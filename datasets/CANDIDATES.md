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

## Open questions before building

- FAIR-CSAR slice size and how many slices: the instruction PDF gives only instance counts (170k SL) and the file naming
  (`..._02048_26208` suggests 2048-px slices with an offset). Confirm after download; decide the SSL working frame
  (SARDet used shorter side 512; at 1 m a 2048 slice downsampled 4x would make a 40 m aircraft 10 px, so we may need to
  tile instead of downsample: this is the S2 tile question again, now for a good reason).
- Chip rule for aircraft: the SARDet rule (window = max(64, 1.5 x longer side), resize to 128) keeps a 40 m aircraft
  at 1 m about 60 px wide inside the 128 chip. Fine. For bridges and tower cranes the boxes are elongated; same rule.
- Label budgets: with ~170k boxes and 22 classes, 1 % is about 1,700 chips (77 per class on average, imbalanced).
  Report mean per-class accuracy as in S1.
- Leakage: SARMAE never saw FAIR-CSAR (clean teacher). SARATR-X-v2 did (do not use it there).

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
