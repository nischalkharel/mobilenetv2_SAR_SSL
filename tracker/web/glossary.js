// Every label, abbreviation and term used in this study, in plain words. Grouped; searchable on glossary.html.
// Keep entries short: what it is, and where it shows up. Add a line whenever a new label appears in a table.
const GLOSSARY = [
  // ---- row labels and study names ----
  {g: "Row labels", t: "A0", a: "ImageNet start, no SSL, baseline", d: "MobileNetV2 with its ImageNet weights, fine-tuned directly on the labeled ATRNet-STAR chips. No self-supervised step. Every other row is compared against it."},
  {g: "Row labels", t: "A1, A2, A3", d: "The first study: MobileNetV2 with SSL on the ATRNet-STAR unlabeled chips, then fine-tuned on ATRNet-STAR labels. A1 = DINO, A2 = LeJEPA, A3 = SparK. 'In-domain' because SSL and test use the same dataset."},
  {g: "Row labels", t: "P4", a: "CondDINO", d: "DINO with per-cell centering keyed by acquisition metadata (which sensor setting, which polarization). It uses metadata fields, not class labels, but that is still information the other rows do not get, so it is kept apart. Its controls: 'scrambled' (metadata shuffled) and '3 fields'."},
  {g: "Row labels", t: "XA", a: "arm A, cross-dataset rows", d: "SSL on SARDet-100K scenes, then fine-tuned and tested on ATRNet-STAR chips. The X means 'cross-dataset', A names the first arm (SARDet pool). Folder names: XA_<method>_sardetssl_mobilenetv2."},
  {g: "Row labels", t: "XI", a: "arm I, in-domain check of RegionDINO", d: "The RegionDINO engine run on the ATRNet-STAR chips themselves (plain DINO reference, both RegionDINO modes, both controls). Asks whether the object views help when the SSL images are already chips. Answer so far: no. Folder names: XI_<method>_atrssl_mobilenetv2."},
  {g: "Row labels", t: "XC", a: "arm C (planned)", d: "SSL on a mixed or new scene pool (for example Umbra or Capella imagery). Not run yet."},
  {g: "Row labels", t: "arm A / arm I / arm C", d: "The three arms of the cross-dataset study: A = SSL on SARDet scenes, I = the same RegionDINO engine on the ATRNet chips (in-domain), C = a new scene pool. Same as the XA / XI / XC prefixes."},
  {g: "Row labels", t: "DINO, RegionSSL engine", a: "XI DINO, DINO reference, suite recipe", d: "Plain DINO run by the RegionDINO code on the ATRNet chips. It should reproduce A1 (it does, within a point) and is the fair reference for the XI RegionDINO rows, because only the views differ."},
  {g: "Row labels", t: "control", d: "A row that keeps everything of a method except the one ingredient being tested. Drawn lighter. If the control gets the same gain as the method, the ingredient did nothing."},
  {g: "Row labels", t: "random windows", d: "The RegionDINO control: crops of the same sizes as the proposed regions, but placed at random spots in the image instead of on the bright regions. Tests whether the gain comes from WHERE the crops are or merely from having small crops."},

  // ---- methods ----
  {g: "Methods", t: "SSL", a: "self-supervised learning", d: "Training the backbone on images with no labels, by making it solve a task built from the images themselves (match two views of the same image, fill in masked patches). The result is a backbone checkpoint, which is then fine-tuned with labels."},
  {g: "Methods", t: "DINO", d: "The SSL method used as the main reference: a student network is trained to match a slowly updated teacher on different crops ('views') of the same image. Two large global views and several small local views per image."},
  {g: "Methods", t: "LeJEPA", d: "A joint-embedding SSL method (predict the embedding of one view from another) with a regularizer that keeps the embeddings spread out. Second SSL route in the first study."},
  {g: "Methods", t: "SparK", d: "Masked image modelling for convolutional networks: hide patches of the image and train the backbone to reconstruct them. Third SSL route. Transfers best across datasets among the plain methods."},
  {g: "Methods", t: "CondDINO", a: "P4", d: "DINO whose centering (the running mean subtracted from the teacher output) is kept per metadata group instead of globally. See P4."},
  {g: "Methods", t: "RegionDINO", a: "RegionSSL", d: "Our method. A label-free pixel rule proposes bright compact regions in each unlabeled image; DINO views are then cut around those regions so the backbone learns objects rather than background. Two modes: scene+object (keeps the normal whole-scene views and adds object views) and object-only (object views only)."},
  {g: "Methods", t: "object-only", d: "RegionDINO mode where every DINO view is cut around a proposed region; no whole-scene views. The mode that works (cross-dataset +9.4 points at 1 % labels over plain DINO)."},
  {g: "Methods", t: "scene+object", d: "RegionDINO mode that keeps DINO's whole-scene views and adds object views. Gains about a third of object-only, and its response maps stay scene-level."},
  {g: "Methods", t: "region rule / proposer", d: "The label-free rule that finds regions: Felzenszwalb superpixels brighter than the image median, grouped into boxes. Runs once per pool image; stored in regions.npz."},
  {g: "Methods", t: "MobileNetV2", d: "The compact backbone (2.2 M parameters in the feature extractor) used everywhere. Starts from TorchVision's IMAGENET1K_V2 weights."},
  {g: "Methods", t: "backbone", d: "The feature-extracting part of the network, without any task head. SSL produces a backbone checkpoint; fine-tuning adds a classifier or detector head on top and trains everything."},
  {g: "Methods", t: "fine-tune", d: "Supervised training on the labeled set, starting from a backbone (ImageNet or SSL). All layers train. Selection: best validation accuracy (classification) or best val-select mAP (detection)."},
  {g: "Methods", t: "frozen backbone", d: "A backbone used as is, no training. The response maps are measured on frozen backbones so they show what SSL learned, not what fine-tuning added."},
  {g: "Methods", t: "ImageNet start", a: "A0, no SSL, baseline", d: "The starting weights of every row: MobileNetV2 trained on ImageNet photos. 'No SSL' rows go straight from here to fine-tuning."},

  // ---- data ----
  {g: "Data", t: "ATRNet-STAR", a: "SOC-40", d: "Vehicle classification dataset of SAR chips: 128-px images each centred on one vehicle, 40 classes. 'SOC-40' is its standard operating-condition 40-class subset. Used for in-domain SSL (unlabeled pool), fine-tuning and the main test (29,284 test chips)."},
  {g: "Data", t: "SARDet-100K", d: "Detection dataset of full SAR scenes (about 117,000 images, shorter side resized to 512 here) stitched from nine source datasets. Six classes: ship, aircraft, car, tank, bridge, harbor. Used as the cross-dataset SSL pool (94,034 scenes) and for the detection task."},
  {g: "Data", t: "chip vs scene", d: "A chip is a small image cut around one object (ATRNet-STAR). A scene is a full image where objects are small and background dominates (SARDet-100K). The difference is the whole point of RegionDINO: on scenes plain DINO learns background."},
  {g: "Data", t: "unlabeled pool", a: "SSL pool", d: "The images SSL trains on, with labels ignored. Validation and test images are excluded. Pools: ATRNet-STAR chips or SARDet-100K scenes."},
  {g: "Data", t: "labeled set / label budget", a: "1 %, 3 %, 10 %", d: "The subset of training labels used for fine-tuning: 1, 3 or 10 % of the training set, drawn once per split and reused by every method. Smaller budgets show the value of SSL most."},
  {g: "Data", t: "split", a: "s1 / s2 / s3", d: "One of three repetitions. Each split has its own labeled subsets and its own SSL seed; fine-tune seeds are fixed. Three splits give the dots in the charts and the (sd) in the tables."},
  {g: "Data", t: "cell", d: "One (method, label budget, split) combination = one fine-tune run. A method has 9 cells. 'done 9/9' means every cell has a completed run."},
  {g: "Data", t: "official test set", d: "The dataset's own test partition, scored once per run with the selected checkpoint. Never used for selection."},
  {g: "Data", t: "val_select", d: "SARDet detection only: an 800- to 2,000-image slice of the validation set used to pick the best epoch quickly, instead of scoring the full 10,492-image validation set each time."},
  {g: "Data", t: "source dataset", a: "HRSID, SSDD, MSAR, OGSOD, SADD, SAR-AIRcraft, SIVED, AIR_SARShip, Shipdataset", d: "The nine datasets SARDet-100K was assembled from. Per-source tables split results by them because target size and sensor differ a lot (OGSOD targets are tiny, SIVED's large)."},

  // ---- metrics ----
  {g: "Metrics", t: "accuracy", d: "Share of test chips classified correctly, in %. The ATRNet-STAR metric. Mean-per-class accuracy is also stored but the tables show overall accuracy."},
  {g: "Metrics", t: "mAP", a: "mAP@[.5:.95], AP50, AP75", d: "Detection quality: average precision over the six classes, averaged over box-overlap thresholds from 0.5 to 0.95 (the COCO rule). AP50 uses only the 0.5 threshold and is more forgiving. Shown as 0.xxx, or 'x100' as points."},
  {g: "Metrics", t: "paired difference", d: "Method minus reference on the SAME split and labeled set, then averaged over splits. Removes the split-to-split noise. 'Positive on 9 of 9 cells' means every split and budget agreed."},
  {g: "Metrics", t: "~ (tilde)", d: "In the difference charts: the splits disagree in sign, so the mean is not a reliable direction."},
  {g: "Metrics", t: "response map", a: "saliency map", d: "Where a frozen backbone responds in an image: the channel norm of its last feature map, upsampled to the image. Measured against the annotation boxes with no labels used for training."},
  {g: "Metrics", t: "mass in box", d: "Share of a response map's total that falls inside the annotated box(es). A map with no preference scores the box area share (17.2 % on ATRNet chips, 3.4 % on SARDet scenes)."},
  {g: "Metrics", t: "peak in box", d: "Share of images whose single strongest response pixel lies inside an annotated box. The clearest 'is it looking at the object' number."},
  {g: "Metrics", t: "concentration", d: "Mass in box divided by the box area share. 1.0 = no preference, 2.0 = twice the attention the box's size would give it."},
  {g: "Metrics", t: "uniform map", d: "The dashed reference line in the response-map charts: what a map that spreads evenly over the image would score."},

  // ---- infrastructure ----
  {g: "Infrastructure", t: "CRC", a: "Pitt cluster, h2p", d: "The University of Pittsburgh Center for Research Computing cluster where every real run happens (A100 40 GB GPUs, Slurm queue). The PC only runs smoke tests and this tracker. CRC log timestamps are UTC."},
  {g: "Infrastructure", t: "Slurm job / sbatch / squeue", d: "The cluster's queue: sbatch submits a job script, squeue lists what is pending (PD) or running (R). Job names like xa_atr_dino_s1 encode method and split."},
  {g: "Infrastructure", t: "staging", a: "stage_dataset, stage_atrnet", d: "Copying the dataset from the shared /ix disk to the GPU node's local disk at job start (one tar read). Without it, jobs reading thousands of small files run 3 to 10 times slower."},
  {g: "Infrastructure", t: "run folder", a: "run_info.json, history.csv, eval_test", d: "Every SSL or fine-tune run writes a timestamped folder: config.json (settings), run_info.json (status, timing, commit, checkpoint hashes), history.csv (per-epoch metrics), eval_*/metrics.json (scores). The tracker reads these."},
  {g: "Infrastructure", t: "compare.py", d: "Two scripts that gather run folders into tables: SARDet_to_ATR/compare.py (ATRNet classification rows) and SARDet/compare/compare.py (SARDet detection rows). The tracker reads their outputs."},
  {g: "Infrastructure", t: "commit", d: "The git version of the code a run used, stored in its run_info.json. Lets a result be traced to the exact code."},
  {g: "Infrastructure", t: "smoke test", d: "A tiny run (a few images, one or two epochs) that checks the code path works. Never a result; excluded from every table."},
];
