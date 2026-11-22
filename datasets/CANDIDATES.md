# A harder second dataset: candidates and recommendation (2026-11-20)

Why: on SARDet-100K the chip test is saturated (no SSL already 81 % mean per-class / 92 % overall at 1 % labels,
97 % at 10 %; tracker conclusion c07), so there is little room for any method to show a gain. We want a dataset
where (a) a compact model struggles, (b) a large SAR-pretrained model is known to do much better (so the teacher
in TeacherRegions has something to teach), (c) the images are scenes with boxes, so our scene-pool SSL -> chip /
detection pipeline applies unchanged, and (d) the data were NOT in the teacher's pretraining pool.

## What the big models were trained on (decides "clean" tests)

- SARMAE (our teacher; CVPR 2026) pretrained on SAR-1M = 18 public datasets: MSTAR, OpenSARShip, SEN1-2, SAR-Ship,
  FUSAR-Ship, HRSID, SSDD, SADD, MSAR, SAR-AIRcraft, SIVED, OGSOD, SARDet-100K (+RSAR), M4-SAR, BRIGHT,
  OpenEarthMap-SAR, AIR-PolSAR-Seg. **Not in it: ATRNet-STAR, FAIR-CSAR, SAR-ACD, SRSDD, SDFSD.**
  (Table 6 of arXiv 2512.16635.) So nearly every classic SAR target benchmark is already "seen" by SARMAE.
- SARATR-X-v2 (arXiv 2607.23238; weights on Hugging Face, gated, MIT; repo updated 2026-11-13) is the newest and,
  by its own tables, stronger than SARMAE (ATRNet-STAR SOC-50 linear probe 97.8 vs SARMAE 81.4). But it pretrained on
  ATRNet-STAR (68k) and FAIR-CSAR (24k) among 17 datasets, so it is not a clean teacher for either. SARMAE stays our
  teacher for ATRNet and for FAIR-CSAR.

## Candidates

| dataset | what it is | why hard | size / access | in SARMAE pool? | fits our pipeline? |
|---|---|---|---|---|---|
| **FAIR-CSAR-V1.0 SL** (TGRS 2025, AIRCAS) | Gaofen-3 spotlight 1 m, 175 full scenes cut to large slices, **22 fine-grained classes** (11 aircraft types incl. A220/A320/A330/737/747/767/777, 7 ship types, bridge, tank, tower crane), ~170k oriented boxes, official trainval/test split | Oriented R-CNN (ResNet-50) reaches only **39 mAP@0.5**; aircraft types differ in metres | 250 GB total (float32 SLC .mat + uint8 PNG intensity + DOTA txt + XML); free for research with citation; registration by e-mail on radars.ac.cn (English page) | **no** | **yes**: scenes for SSL, chips from boxes for 22-class classification, detection as in SARDet |
| **SAR-ACD** (TGRS 2022, AIRCAS) | 3,032 Gaofen-3 chips of **6 civil aircraft types** (A220, A320/321, A330, ARJ21, 737, 787), ~500 per class, 60-110 px, HH | ResNet-50 **59.7 %**, Swin 67.5 %, SARMAE 95.1 %, at 30 % labels (SARMAE paper Table 2): exactly "small model struggles, big SAR model wins" | tiny; plain GitHub download (AICyberTeam/SAR-ACD), no licence file | no (but SAR-AIRcraft, same lab, same airports, is in it: possible image overlap) | chips only, no scenes: a second cross-dataset transfer test like ATRNet (XA style), not an SSL pool |
| SRSDD-v1.0 (2021) | Gaofen-3 1 m, 666 slices 1024 px, 2,884 ships in **6 types**, 63 % inshore | O-RCNN 56 mAP; heavy imbalance (2,053 bulk cargo vs 25 law-enforcement) | small; GitHub (HeuristicLU/SRSDD-V1.0), CC BY | no | detection test only; too small for an SSL pool |
| SDFSD-v1.0 (2024) | 846 sub-metre scenes (GF-3, Umbra, Capella, ICEYE), 35,787 slices, 96,921 ships in **15 types** | fine-grained mAP 25-33 | **not released** ("upon formal publication"; nothing online as of today) | no | would be ideal; unavailable |
| OpenSARShip / OpenSARShip 2.0 | Sentinel-1 20 m ship chips, 3-6 usable types | 6-class accuracy ~60 % in the survey | free, SJTU | **yes** | chips only; teacher has seen it |
| FUSAR-Ship | GF-3 ship chips, 15 types (3 make 81 %) | imbalance | free | **yes** | chips only; teacher has seen it |
| OpenSARUrban, So2Sat LCZ42, BigEarthNet-S1 | Sentinel-1 land-cover / urban patches | yes, but not targets | free | partly (SEN1-2) | off-topic for the sponsors (no objects) |

Sentinel-1 scenes + OpenSARShip chips (the earlier plan): still possible for a sensor-matched test, but OpenSARShip
is inside SAR-1M, 20 m resolution gives 3-6 usable classes, and the scenes would have to be downloaded and matched
to the chips by hand. FAIR-CSAR gives all of that in one package at 1 m with 22 classes.

## Recommendation

1. **FAIR-CSAR-V1.0 SL as the second scenes-in / chips-out dataset.** It is the only released dataset that is hard
   (39 mAP for a ResNet detector; 22 classes), large (170k boxes), scene-based, and unseen by SARMAE. One download gives
   the SSL pool (PNG intensity slices, labels unused), the chip classification test (chips cut from the oriented boxes,
   same rule as SARDet_chips), and a detection test. Expected: no SSL far below 90 % on 22 classes, so SSL gains have room.
   Cost: registration (name, affiliation, e-mail; Nischal does this), 250 GB download (the PNG + annotation part alone is
   much smaller than the SLC .mat part; take the PNGs first), a new `paths` entry and staging on /ix.
2. **SAR-ACD right away as a cheap third transfer test (chips).** 3,032 chips, downloadable now, where ResNet-50 gets 60 %
   and SARMAE 95 %. Fine-tune every existing SARDet-SSL backbone on it at 1 / 3 / 10 % like the ATRNet transfer (XA) and
   add a row block "XC" to the tracker. Caveat for the paper: SAR-AIRcraft (same airports) is in SAR-1M, so the teacher
   may have seen near-duplicates; it is not in anything we train.
3. Keep SRSDD on the list as a small inshore-ship detection test if we need a third detection dataset; skip the
   land-cover sets.

## FAIR-CSAR-V1.0 SL as it really is (inventory 2026-11-22, `fair_csar_inventory.py` over every annotation file)

Both SL archives are on CRC, verified (md5), extracted without the complex .mat files to
`/ix/ageorge/nik96/data/FAIR_CSAR/extracted/{test,trainval}/{PNGImages,Annotations,METAXmls,KeyPoints}` (12 GB).
The annotation and metadata folders plus 20 test PNGs are on the PC as `..\2026_2027\FAIR_CSAR_sample\` (not in git).
Full tables: `datasets/outputs/fair_csar_inventory.md`.

- Slices are 1024 x 1024 px, 8-bit grey PNG (uint8, mean about 30 to 55, long bright tail), 1 m nominal, HH mostly.
  The SLC name fields are range and azimuth offsets, not a 2048 grid as guessed earlier.
- trainval: 105 scenes, 11,959 slices, 69,298 objects. test: 26 scenes, 2,706 slices, 15,987 objects. Every slice
  holds at least one object (median 3, p95 21): the dataset ships only annotated slices, so the SSL pool is "scenes
  with objects", like SARDet.
- 22 classes, all present in both splits: 12 aircraft (A220, A320, A330, 737, 747, 767, 777, Fokker-50, Gulfstream,
  Helicopter, Airfreighter, Other_Aircraft), 7 ship (Bulk_Carrier, Container_Ship, General_Cargo_Ship,
  Motion_Defocusing_Ship, Oil_Tanker, Other_Ship, Warship), Bridge, Tank, Tower_Crane. Imbalanced: Tank 15,364 and
  General_Cargo_Ship 13,196 in trainval against Boeing747 332 and Fokker-50 487.
- Object size (oriented-box long side, slice pixels): median 52, p5 18, p95 158. Aircraft 33 to 78 px median
  (Helicopter 14), ships 51 to 182, Bridge median 84 with a tail to 850, Tower_Crane 91. 44 % of objects fall in 32
  to 64 px, 30 % in 64 to 128.
- 1 % of trainval per class: 3 (Boeing747) to 154 (Tank) chips; six classes get under 10. A per-class floor
  (for example at least 5) or a 2 % smallest budget is needed; mean per-class accuracy is the headline, as in S1.
- Site overlap: many scenes are the same harbour on different dates (14 scenes at E139.7 N35.5, 12 trainval + 2
  test), so the official split separates acquisitions, not places. Fine for the chip test (that is the benchmark's
  own split); state it in the paper.

## Protocol decision for the FAIR-CSAR chip study (proposed, to review before building)

- SSL pool: the 11,959 trainval slices only (test slices never enter SSL). Working frame, two options to run as
  arms: (a) tiles, each slice cut 2 x 2 into 512-px tiles at native 1 m (47,836 tiles; keeps the detail that makes
  this dataset harder; matches the suite's 512 frame); (b) the slice downsampled to 512 (2 m per pixel, 11,959
  images; a 50-px object becomes 25 px, like SARDet). (a) is the main arm; (b) answers the sponsors' resolution
  question from 2026-11-13.
- Chips: the SARDet_chips rule unchanged (window = max(64, 1.5 x long side), resize to 128), cut at native
  resolution from the oriented box's axis-aligned extent; difficult-flag objects kept but reported separately.
- Labels: 22 classes, mean per-class accuracy on test, budgets 1 / 3 / 10 % of trainval chips with a per-class
  floor of 5, three splits as in S1.
- Rows: no SSL, DINO, object-only RegionDINO + random control (and the teacher pair if the S1 teacher jobs say
  anything), each on arm (a); DINO and object-only on arm (b). SARMAE linear probe / fine-tune as the big reference
  (clean: never trained on FAIR-CSAR).

## Sources

- SARMAE paper (SAR-1M table, downstream tables): https://arxiv.org/html/2512.16635v2
- SARATR-X-v2 paper and repo: https://arxiv.org/html/2607.23238 , https://github.com/waterdisappear/SARATR-X-v2 ,
  https://huggingface.co/waterdisappear/SARATR-X-v2
- FAIR-CSAR instruction PDF and data page: https://radars.ac.cn/web/data/getData?dataType=FAIR_CSAR_en&pageType=en ;
  TGRS paper doi 10.1109/TGRS.2024.3519891; ScienceDB doi 10.57760/sciencedb.radars.00019;
  detection baselines from arXiv 2603.14886 (PASTE) Table on FAIR-CSAR
- SAR-ACD: https://github.com/AICyberTeam/SAR-ACD (SCAN, TGRS 2022, doi 10.1109/TGRS.2022.3166174)
- SRSDD-v1.0: Remote Sensing 13(24):5104, https://github.com/HeuristicLU/SRSDD-V1.0
- SDFSD-v1.0: Remote Sensing 16(21):3952 (data availability: on formal publication; not found online 2026-11-20)
- Ship-classification survey (OpenSARShip / FUSAR-Ship difficulty): https://arxiv.org/html/2503.11906v1
