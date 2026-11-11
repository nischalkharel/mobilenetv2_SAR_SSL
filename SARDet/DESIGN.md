# SARDet-100K detection suite: experiment design (v2 after protocol review, 2026-11-11)

Goal: repeat the ATRNet-STAR SSL comparison (supervised baseline, DINO, LeJEPA, SparK, and DINO with
per-condition-cell teacher centering) on SAR *object detection* with the same compact backbone (MobileNetV2),
with a protocol that is identical across methods and audited for leakage. v1 was reviewed by a red-team agent;
the changes it caused are marked [review].

## 1. Dataset facts (measured on the local copy, 2026-11-11)

- SARDet-100K (Li et al., NeurIPS 2024). COCO-format boxes, 6 classes: ship, aircraft, car, tank, bridge, harbor.
- Official splits (by image): train 94,493 / val 10,492 / test 11,613 (116,598). Instances 198,747 / 22,703 / 24,203.
- Images: 8-bit grey (stored mostly as RGB with identical channels) JPEG/PNG/BMP of varied size: 256x256 (62 %), 512x512 (15 %), 800x800, 224x224,
  1000x1000, 500x~330. 61 % of boxes are "small" (< 32^2 px at native size). No `ignore`/`iscrowd`/segmentation.
- The release folder on this PC had only `train.json` and `val.json` extracted; the archive also holds
  `test.json` (24,203 test annotations) and `mapping.json` (original file name -> anonymised name). Both were
  extracted into the dataset folder (`SARDet_100K/test.json`, `SARDet_100K/mapping.json`) and must travel with it.
- Per-image metadata: the annotation JSON has none; the ORIGINAL file name (via mapping.json) gives
  - the source dataset: Shipdataset 39,729 / MSAR 30,158 / SAR-AIRcraft 18,888 / OGSOD 18,331 / HRSID 5,604 /
    SSDD 1,160 / SIVED 1,044 / SADD 883 / AIR_SARShip 801;
  - PER IMAGE only for Shipdataset: satellite (Sentinel-1 / Gaofen-3) and polarisation (HH/HV/VH/VV); for SIVED the
    sub-source (MSTAR / FARAD). Every other source has ONE sensor bundle per source from the paper's Table 2
    (AIR_SARShip GF-3 C VV; MSAR HISEA-1 C; SADD TerraSAR-X X HH; SAR-AIRcraft GF-3 C; OGSOD GF-3 C), or a
    MIXTURE with no per-image field (HRSID: S-1B / TerraSAR-X / TanDEM-X; SSDD: S-1 / RadarSat-2 / TerraSAR-X).
  - for the sliced sources (SAR-AIRcraft, AIR_SARShip-1: 512 px patches with 200 px overlap; HRSID: 800 px tiles
    of 136 scenes) the PARENT image / scene id. MSAR, OGSOD, SSDD, Shipdataset, SADD, SIVED carry no parent id.
- Sources map almost one-to-one onto classes (OGSOD = bridge/harbor/tank, SADD and SAR-AIRcraft = aircraft,
  SIVED = car, HRSID/SSDD/AIR/Shipdataset = ship, MSAR = mixed). "Source" is a curation artefact, not a sensor
  field; consequences for CondDINO in section 5.
- Leakage inside the official release (MD5 of file bytes): 5,701 duplicate groups; 303 straddle official splits.
  Neighbouring patches of one parent share up to 40 % of their pixels. [review] Near-duplicates that are
  re-encoded or re-cropped from one scene in the sources WITHOUT a parent id (MSAR, OGSOD, SSDD, Shipdataset)
  cannot be detected and are a stated limitation of the benchmark that affects every method equally; the audit
  also counts val-test straddling groups.

## 2. Subsets

Drawn from the OFFICIAL TRAIN only. Official val and official test are never re-split.

- Unit of drawing = "group": images that share a parent image OR identical content (file MD5; [review] plus the
  MD5 of the decoded pixels, which catches lossless re-encodings) are one group (union of the relations) and
  always land in the same subset. Groups with a member in official val or test are EXCLUDED from every
  training-side subset, so no training image is a copy or patch-neighbour of a scored image.
  Parent = the large image of the 512 px patch sources (SAR-AIRcraft, AIR_SARShip-1; 200 px = 40 % overlap).
  HRSID's scene id is RECORDED but not a parent: its 800 px tiles overlap <= 25 %, HRSID's own split mixes
  scenes, and grouping by scene removed 3,638 of its 3,642 train tiles in the first draw (audit 2026-11-11),
  i.e. the whole source. With the final rule 459 train images are excluded (SAR-AIRcraft 395, Shipdataset 50,
  SSDD 12, HRSID 2); 132 of HRSID's 136 scenes straddle the official splits, reported as a benchmark property.
- SSL pool = official train minus the excluded groups = 94,034 images; the SAME pool for all splits (val is
  fixed, so nothing else varies in the pool). [review] Labeled sets are drawn INSIDE the SSL pool, exactly as on ATRNet-STAR
  ("the labeled set is entirely inside the SSL pool"); SSL uses no labels, so this is the standard protocol.
- Labeled sets per split s in {1,2,3} (seed = s-1): 1 %, 3 %, 10 % of official train (quota 945 / 2,835 / 9,449;
  actual on split 1: 962 / 2,841 / 9,460 because whole groups are added),
  NESTED (1 % inside 3 % inside 10 %), stratified by source: whole groups are added per source until that
  source's quota is reached (tolerance: the last group may overshoot), so actual counts differ slightly from the
  quota and are recorded per source per split. For the sliced sources the 1 % set is a handful of parent scenes.
- Validation (per-epoch model selection): [review] `val_select` = a fixed 2,000-image subset of official val,
  stratified by source, seed 0, the same for every run, manifest-hashed. The selected checkpoint (and the
  last-epoch checkpoint, for selection sensitivity) are then scored once on the FULL official val and once on
  official test.
- Test (final numbers only, never used for any decision): official test, as released. Because every group
  touching val/test is already excluded from training, the full official test is clean for THIS study;
  `test_clean` (official test minus images with a twin/parent-sibling in official train) is reported per source
  only for comparison with published SARDet-100K numbers trained on the full train set.
- What the three splits vary: the labeled draw (1/3/10 %) and the SSL seed (one SSL run per split, seed = s-1).
  The SSL pool, val and test are fixed. The fine-tune seed is 0 everywhere. Comparisons are PAIRED within a split.
- Every subset is a CSV manifest (file name, official split, image id, size, source, satellite, polarisation,
  band, group; NO labels) with the SHA-256 recorded in every run record; `audit_splits.py` asserts pairwise
  disjointness, nesting, exclusion of val/test groups, and prints the leak counts above.

## 3. Detector (identical for every method; built by the shared package from a backbone state dict)

- Backbone: torchvision MobileNetV2 `features` (ImageNet-1k V2 weights for the baseline; the SSL-exported
  `features.*` state dict for the others; every SSL method starts from the same ImageNet file, as on ATRNet).
  [review] BatchNorm in the backbone is TRAINED during fine-tuning for every method (as on ATRNet); nothing is
  frozen. Which weights each SSL method exports: DINO/CondDINO the EMA teacher, LeJEPA and SparK the encoder;
  always the FINAL SSL epoch, never a selected one.
- Neck: FPN (torchvision `BackboneWithFPN`) on strides 4/8/16/32 (features[3], [6], [13], [18] = 24/32/96/1280
  channels), 128 channels, plus a max-pooled P6.
- Head: torchvision Faster R-CNN (RPN + RoIAlign + 2-fc box head), 6 classes + background.
  Anchors [review, from the box audit]: sizes (8,16) (16,32) (32,64) (64,128) (128,256) on P2..P6, ratios
  0.5/1/2; after the 512 resize only 2.7 % of train boxes have no anchor with IoU >= 0.5 (16-256 would miss 6.9 %).
  Scoring settings fixed explicitly: box_score_thresh 0.05, box NMS 0.5, 100 detections per image (= COCO
  maxDets), RPN top-n 2000/2000 train, 1000/1000 test.
- Input: `GeneralizedRCNNTransform` min/max side 512 (256 px images up-scaled x2, 800/1000 px down-scaled);
  1-channel images replicated to RGB, ImageNet mean/std.
- Fine-tune schedule (same for all methods, per budget): AdamW, one lr for the whole network 2e-4, weight decay
  0.05, 1-epoch linear warm-up then cosine to 1 % of lr; batch 8; epochs 1 %: 60, 3 %: 40, 10 %: 24; fp32
  (PyTorch default TF32 flags, recorded). [review] The shared lr is checked ONCE, BEFORE any real fine-tune,
  with `crc/lr_check.sbatch` (1e-4 / 2e-4 / 5e-4 on val_select, 10 %, split 1). Decision rule, fixed in advance:
  the BASELINE check decides the value (so no SSL backbone can tune the protocol in its own favour); the same
  check on one SSL backbone, run later, is reported as a robustness check and cannot change the value. The
  chosen lr and the lr-check run folders are written into `shared/finetune_config.json` (`_lr_decision`).
- Augmentation: horizontal flip 0.5 only [review: vertical flips were rejected on ATRNet because they move the
  SAR shadow/layover geometry; same rule here].
- Seeds [review]: fine-tune seed 0 for every run; `torch.manual_seed` is called right before the FPN/RPN/box head
  are built and again before the loader, so head initialisation and data order are identical across backbones
  within a split (the smoke test asserts identical head tensors for two different backbones).
- Model selection: val_select mAP@[.5:.95] every 2 epochs; best = highest; `best.pt` and `last.pt` kept.
- "Exact resume" means state-exact: same batches, schedules, RNG streams, optimizer, centers and counters; the
  numbers can differ at the cuDNN-benchmark / mixed-precision level between a resumed and an uninterrupted run.

## 4. SSL (one run per split per method; equal epochs)

- Pool: section 2 (same for all splits; seed = split-1).
- [review] View recipe, ONE shared block for all SSL methods (`shared/ssl_common.json`): the image is first
  resized by the detector's rule (512), so object scale matches detection; views are crops taken at that scale
  with zoom jitter 0.75-1.33 (window = view size x zoom, then resized to the view size): global views 224 px,
  local views 96 px; horizontal flip 0.5; brightness/contrast jitter 0.4 (p 0.8); Gaussian blur as in the ATRNet
  configs; no solarisation (inverts scatterers). Per method (as on ATRNet): DINO 2 global + 8 local; LeJEPA 8
  global; SparK 1 global view with a 60 % patch mask. `view_stats.py` reports the fraction of views that contain
  at least one ground-truth box (labels read only for this diagnostic).
- `epochs` is ONE value in `shared/ssl_common.json` (default 100, as on ATRNet). [review] "Equal epochs" is the
  stated control; GPU-hours and total view-forwards are logged per run, and the ATRNet lesson that equal epochs
  are not equal compute (LeJEPA ~1.9x DINO) is repeated in the README. Every SSL run checkpoints each epoch and
  resumes with `--resume`; the sbatch scripts re-submit themselves with a dependency until the run completes, so
  a 24 h QoS limit never loses work.
- Diagnostics per epoch: loss, teacher entropy / marginal entropy (DINO), feature std and cosine (all),
  reconstruction loss (SparK), SIGReg and agreement terms (LeJEPA), lr, seconds.
- Export: `backbone_epoch<E>.pth` with torchvision `features.*` keys incl. BN buffers; SHA-256 recorded.

## 5. CondDINO_SARDet (the P4 method on this dataset) [review: reworked]

ATRNet-STAR gave scene, band, depression and azimuth per chip, fields that cut across classes. SARDet-100K
gives, per image, only what the source name carries, and the source IS the class for most images. Honest
statement: no per-image acquisition field that is independent of the class exists in SARDet-100K. The suite
therefore runs the mechanism with the fields that exist and measures the leak instead of hiding it:
- `source_pol` (default cells): source dataset, refined by the per-image satellite x polarisation for Shipdataset
  and the sub-source for SIVED (19 cells of 80 to 27,159 images incl. two "unknown" cells for Shipdataset
  names without sensor tokens and SIVED-unknown; cell sizes are recorded, and the ATRNet "rare-cell momentum"
  argument does not apply here). Measured leak on the split-1 pool: 17 of 19 cells are 100 % one class; the
  cell alone predicts the dominant class at 94.7 % (majority 64.9 %, scrambled 64.9 %); I(cell; class set)
  1.22 of 1.56 bits (scrambled 0.002). `cell_report.py` prints the class-vs-cell contingency,
  purity and I(cell; class) so the reader sees how much of the cell is class information. This is a condition
  in the sensor sense (resolution, band, polarisation bundle) but NOT deployable and NOT class-independent.
- `perm`: the same cells with the image-to-cell assignment scrambled (size-matched control).
- `oracle_class`: cells = the set of classes present in the image (labels used; an explicit label-leak CEILING,
  never a method). If `source_pol` ~ `oracle_class` >> `perm`, the gain is class information, not condition.
- `radiometric`: label-free, deployable cells computed from the image itself (k-means, k = 16, on a 32-bin
  intensity histogram + log image size, fitted on the SSL pool): the closest honest substitute for a
  per-image acquisition condition, available at deployment. Measured leak: 16 cells, majority vote 73.0 %
  (scrambled 64.9 %), I(cell; class set) 0.52 bits, 5 cells > 90 % one class: about a third of source_pol's.
  `oracle_class` (13 cells) reaches 99.9 % and the full 1.56 bits by construction.
- Dropped: `sensor_band` (v1), because sensor/band are not per-image for 7 of 9 sources (removed from the code too).
Cells are used only inside the DINO loss during SSL; the detector never sees them.

## 6. Metrics, charts, analyses

- COCO mAP@[.5:.95], AP50, AP75, APs/APm/APl, per-class AP and AP50, [review] per-SOURCE AP (ship AP across
  HRSID / SSDD / Shipdataset / AIR separates sensor from class), PR curves at IoU 0.5 per class, on val and test
  (full val, test, test_clean), written as JSON + CSV + PNG; detections saved so an image-level bootstrap CI
  of test mAP can be computed afterwards (`evaluate.py --bootstrap N`).
- Training curves (loss components, lr, val_select mAP), SSL curves + collapse diagnostics.
- Frozen-feature probes on each exported backbone (pooled features of official val): linear probe for class
  presence (multi-label), decodability of source / satellite / polarisation (condition), kNN agreement; the same
  feature-std / rank diagnostics on the fine-tuned backbones [review].
- Visuals: backbone response map (channel norm of the last feature map) and FPN P3 response over detections;
  prediction-vs-ground-truth panels (one per source dataset) for every fine-tuned model.
- Model cost, once: parameters, FLOPs at 512 px, CPU latency of the full detector (same for all methods).
- `compare.py` [review]: per-split values; per-split PAIRED differences (method minus baseline) with mean, range
  and sign consistency across the 3 splits; mean and std of the raw values for reference; no t-tests at n = 3;
  SSL GPU-hours per method; tables (markdown + CSV) and charts; one command.
- Fine-tune seed noise: `crc/finetune_repeat.sbatch` re-runs baseline 10 %, split 1 with seed 1 (one number
  quoted next to the method gaps, as the "~1 pt" hardware-noise figure was on ATRNet).

## 7. Run records and machines

- Every output folder: config, git commit, manifest hashes, dataset root, hostname, GPU, torch/torchvision/
  pycocotools versions, TF32 flags, wall time, peak GPU memory, start/end time; SHA-256 of the backbone used.
- Paths: `shared/paths.py` reads environment variables with PC defaults; `crc/env.sh` sets them on CRC.
- Smoke test per experiment: a few dozen images, 1 epoch, end to end incl. evaluation and plots.
- CRC: one sbatch per experiment, A100, same conventions as mobilenetv2_SAR_SSL/crc; SSL jobs self-resubmit.
