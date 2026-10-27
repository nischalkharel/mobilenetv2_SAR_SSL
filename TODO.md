# To do after the current A1 run

Proposed work, not done yet. Started 2026-10-14 while the first full DINO run (`20261014-101106_A1_dino_ssl_mobilenetv2_seed0`) was training. Do not change code that a running job uses.

## 1. Fine-tuning accuracy across SSL epochs (Nischal's request)

Goal: see how the downstream accuracy changes with SSL training length, by fine-tuning the backbone saved at epochs 5, 10, 15, ..., 100 (20 checkpoints).

- [ ] `train_ssl.py`: add a setting, e.g. `"export_every": 5` in `config_ssl.json`, that exports the **teacher backbone** at every 5th epoch (`backbone_teacher_epoch005.pth`, ...). Backbone files only (about 9 MB each, about 180 MB in total). Do not keep every full `checkpoint.pth` (162 MB each).
- [ ] Decide whether resume must refuse a changed `export_every` (it does not change training, so probably allowed, but record it).
- [ ] A small script to fine-tune and score all 20 backbones on validation in one go, with one results table and a chart (validation accuracy against SSL epoch). About 3 min per fine-tune, so about 1 hour.
- [ ] Rerun the smoke test after the change.
- [ ] Then a new SSL run with the exports (about 7.5 h), or decide whether one is needed at all.

Decide before running (research decisions for Nischal):
- [ ] **Analysis or selection?** If the curve is only for analysis, keep the pre-set rule (epoch 100) for A1's reported result. If the best epoch is *chosen* on validation, then the validation labels become part of A1's tuning budget, which must be reported. A0 had no such choice, so say that in the comparison. The test set is never used for this choice.
- [ ] **Epoch 50 of a 100-epoch run is not a 50-epoch run.** With the cosine schedule, at epoch 50 the learning rate is still high and the teacher momentum has not reached 1. The curve shows "partway through a 100-epoch schedule". If a shorter schedule looks just as good, confirm it with a real run of that length.
- [ ] Repeat with more SSL seeds before drawing conclusions from small differences between epochs.

## 2. Smaller items

- [ ] The printed "max 8.32" next to marginal entropy is misleading: it is measured over 128 teacher outputs per batch, so its practical ceiling is ln(128) = 4.85. Fix the printout and the README note.
- [ ] Delete the empty folder `outputs/ssl/20261014-100704_A1_dino_ssl_mobilenetv2_seed0` (a Ctrl+C before training; no results). Waiting for Nischal's OK.

## 3. A2 (LeJEPA)

- [x] Main A2 implemented, smoke-tested and piloted (`Experiments/mobileNetv2_LeJEPA`, 2026-10-16).
- [ ] Full A2 SSL run (100 epochs, about 14 h; started 2026-10-16), then fine-tune from the final backbone, validation, and (when frozen) test.
- [ ] Accuracy vs SSL epoch for A2 from the 5-epoch exports (analysis only).
- [ ] Possible later A2 variants (one at a time, chosen on validation): projector output 64 (paper's ViT ablation), lambda, batch (gradient checkpointing for 64+).
- [ ] Separate controlled experiment: LeJEPA with DINO's crops (2 x 224 + 8 x 96, scales as A1) and 100 epochs.

## 4. Open from earlier sessions

- [ ] Local crop scale ablation: 0.05 (original DINO) against our 0.15, chosen on validation, if compute allows.
- [ ] Repeat seeds for A0 and A1 once the protocol is fixed.

## 5. From the NG meeting (2026-10-19; Joey, Brittany, Abby; joint with Dikchhya)

Asked of Nischal:
- [ ] Results as a side-by-side **bar chart** (A0/A1/A2, later A3), not only a table; add test accuracy once settings are frozen.
- [ ] Implement **SparK (A3)** and run it on CRC.
- [ ] **Seeds re-draw the data split** (labeled set + SSL pool), not only the training seed (Joey). Re-apply the leakage control for each new split.
- [ ] **Label budgets spread wide**: e.g. 1%, 3%, 10%.
- [ ] **Cost of the gain**: SSL training time / GPU-hours / peak memory per method (have: DINO 469.6 min 9.19 GB, LeJEPA 633.7 min 19.8 GB on A100); inference cost is the same backbone for all. Propose the cost metrics on a slide next meeting and ask for feedback.
- [ ] **Second architecture**: collect SHREC lab backbone studies (Pitt and other SHREC sites); candidates EfficientNet (NG has good SAR results with it), MobileNetV3, ResNet. Short list for next meeting.
- [ ] CPU latency: low priority (same MobileNetV2 for every method). If done: Intel server-grade CPUs, roughly 24-50+ cores, inference only.
- [ ] Look at **SARDet-100K** (Brittany, in chat): COCO-style superset of SAR sets (ships, aircraft, ground, MSTAR), larger scenes; possible bigger/more varied SSL pool.
- [ ] Check SOC-40 vs MSTAR target overlap (said "I can do that cross-check"); SOC-50 = SOC-40 + MSTAR (different sensors) not used.
- [ ] Brittany's ATR interests (raised in Dikchhya's slot, apply here too): generalization across depression/aspect angle, band (X vs Ku) and polarization, and background/scene. ATRNet-STAR has EOC splits for these; currently all bands and 4 polarizations are mixed together.
- [ ] Brittany's concern: 128 px chips upscaled to 224 and cropped (little scene context). Explain upscaling loses no information, crops do drop context; keep in mind when choosing the dataset.
- [ ] **Main goal (not later):** novelty for the conference paper, e.g. a method combining ideas from DINO / LeJEPA / SparK. See section 6.

Follow-up email with the slides: confirm DINO's final network is the teacher (EMA of students), and that the student sees all 10 crops (2 global + 8 local) while the teacher sees only the 2 global ones.

Logistics: biweekly on Fridays, joint with Dikchhya, 1-hour series; SHREC conference January 2027, Albuquerque.

## 6. Novelty candidates (keep updating; check literature before claiming)

Joey asked whether adapting SSL to MobileNetV2 is novel: done before, but not in a controlled way. That alone is weak; we need something new.

- [ ] **Physical views as positive pairs:** ATRNet-STAR has the same vehicle, angle and scene in 4 polarizations, 2 bands and repeat passes. Use these as SSL "views" instead of (or with) random crops. We currently remove them only near validation images (leakage control); inside the SSL pool they are free, label-free positives.
- [ ] **Neighbouring aspect angles as positives** (same target, small rotation in azimuth): teaches the angle robustness Brittany cares about.
- [ ] **DINO + SIGReg:** add LeJEPA's SIGReg to DINO (or replace centering/sharpening).
- [ ] **LeJEPA with DINO's multi-crop** (already in section 3): DINO was better and cheaper here; is that because of the small crops?
- [ ] **Masked + embedding:** SparK-style masking as one of the views in DINO/LeJEPA.
- [ ] **SAR-specific augmentations** (speckle-aware noise, dB/contrast changes) instead of the photo ones (solarization is already off for SAR reasons).

## 7. EOC (operating-condition shift) study, from the 2026-10-20 overlap check

Check script: scratchpad `eoc_check.py` (sha256 of every .tif). Findings:
- Every EOC image is an SOC-40 image (0 new images); for each experiment, EOC train + EOC test = all 97,375 SOC-40 images, re-sorted by condition.
- Train condition per experiment: depression 15 deg (test 30/45/60); polarization HH (test HV/VH/VV); band X (test KU, folder `test`); azimuth mostly 0-59 deg (test in 60-deg sectors); scene sandstone + bare soil (test factory, urban (X band only), woodland (11 classes only)). All 40 classes otherwise.
- About 2/3 of every EOC test set is in our current SSL pool, about 30% is SOC-40 test. So the current A1/A2 backbones have seen (unlabeled) most EOC test images: do not use them for EOC claims.
- [ ] Design (proposed): per EOC experiment, split each test condition by acquisition group into a half that may go into the SSL pool (unlabeled) and a half kept only for evaluation; labels = small budget (1/3/10%) from EOC train only. Strict ablation: SSL on EOC train only (about 20-28k images).
- [ ] Priority: depression and polarization (Brittany), then scene; band and azimuth later.

## 8. A3 SparK (implemented 2026-10-27) and a precision finding

- [x] SparK for MobileNetV2 in `Experiments/mobileNetv2_SparK` (authors' code vendored, import lines only changed; smoke test 66 checks pass on the PC GPU). See its README and SPARK_CHANGES.md.
- [ ] Full run on CRC: `sbatch crc/spark.sbatch` (SSL 100 epochs, then 3% / 1% / 10% fine-tunes with validation + test). A100 time not measured yet (expected roughly 12-17 h in float32).
- [ ] **Precision pitfall (possible paper finding; check before claiming):** at SparK's start, the gradient vs a float64 reference has about 100% error with TF32 (PyTorch's default on A100/RTX 30/40), 129% with bf16, 11% in true float32 on the GPU, 3% on the CPU. Likely cause: ImageNet normalization maps the dark SAR background to about -2.1 and BatchNorm follows every conv, so gradients are small remainders of large cancelling sums. A3 runs in true float32 for this reason.
  - [ ] Measure the same for A1 (DINO, fp16 autocast) and A2 (LeJEPA, bf16) at initialization and later in training. If large there too, their results were trained with mostly-noise encoder gradients; rerun in float32 and compare.
  - [ ] Possible fix to test: normalize SAR inputs with the SAR pool's own mean/std instead of ImageNet's (removes the large constant offset); keep everything else equal.
