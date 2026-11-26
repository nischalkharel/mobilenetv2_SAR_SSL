# RegionSSL: label-free "things first" self-supervised learning on SAR scenes

Question: does a compact model learn more useful SAR features when the self-supervised stage is pointed at
the *things* in a scene (found by a pixel rule, no labels) instead of at random crops of the whole scene?
Design and reasoning: [DESIGN.md](DESIGN.md). Everything here is separate from the finished studies: the
detection suite (`SARDet/`) and the ATRNet experiment folders are imported read-only and never modified.

## What is in this folder

| file / folder | what it does |
|---|---|
| `segment_lab.py`, `box_zoom.py` | the lab that chose the region rule (`felz_grouped`) and shows what it finds against the GT boxes |
| `make_regions.py` | runs the rule once over the whole SSL pool and stores each region's box (`regions/regions.npz`, not committed; `regions/regions_summary.json`, committed) |
| `rssl/` | this study's code: `regions.py` (storage), `region_data.py` (scene views + object views), `regiondino.py` (the method), `region_engine.py` (copy of the suite's SSL engine that lets a method bring its own dataset) |
| `RegionDINO_scene_object/` | method 1: DINO on the whole image **plus** DINO on object views (double forward: scene crops and object crops) |
| `RegionDINO_object/` | method 2: DINO on object views **only** (the network never sees a whole-image crop) |
| `*/config_ssl_random.json` | the control for each method: same everything, object views cut around random windows of matched size |
| `crc/make_regions.sbatch` | the one CPU job that makes the region file on CRC |
| `pool_hit_rate.py` | pool diagnostic (2026-11-26, c13): samples plain DINO's own global / local windows over a pool and measures how often they land on bright content (usable regions of the region file; `--gt N` adds the annotated-object check on a sample, labels read only to measure); per source for SARDet-100K; outputs `outputs/pool_hit_rate_<pool>.json` (committed) |
| `rssl/pools.py` | the two unlabeled pools the engine can train on: `sardet` (SARDet-100K scenes, shorter side 512) and `atrnet` (ATRNet-STAR train chips, views from the chip at 224, regions proposed at the native 128) |
| `*/config_ssl_atr.json`, `*/config_ssl_atr_random.json` | arm I: the same two methods and controls on the ATRNet pool (in-domain), run names `*_atr_*` |
| `DINO_reference/` | plain DINO of this engine (scene views only) on the ATRNet pool: the paired reference for the `*_atr` rows; its gap to A1 measures the crop-recipe difference |
| `chip_lab.py` | the region rule on ATRNet chips against the vehicle box (lab check, labels for scoring only) |
| `scene_maps.py`, `crc/scene_maps.sbatch` | response maps of the frozen SARDet-pool backbones on SARDet-100K test scenes against the annotated boxes (the scene counterpart of `SARDet_to_ATR/response_maps.py`); outputs in `outputs/scene_maps/` |
| `crc/ssl_atr.sbatch`, `crc/make_regions_atr.sbatch` | the CRC jobs for arm I (both env files sourced; the SOC-40 chips are staged to the node disk by `stage_atrnet` in the repository's `crc/env.sh`) |

Each method folder has the same four entry points as the suite's methods: `train_ssl.py`, `smoke_test.py`,
`run_finetune.py` (SARDet detection fine-tune), `config_finetune.json`. The exported backbone has the suite's
format, so the detection fine-tune (`SARDet/shared/finetune.py`) and the ATRNet fine-tune
(`SARDet_to_ATR/finetune_atr.py --method regiondino_so | regiondino_o | *_random`) load it unchanged.

## How an object view is made (plain words)

For each image the rule proposes regions (bright compact blobs, grouped). Training picks K of them per image
(bigger ones more often), and for each cuts 2 "object-global" views (the thing with a margin of surroundings,
window = 1.4 to 2.2 x its extent) for the teacher and 4 "object-local" views (a part of the thing, window =
0.5 to 1.0 x its extent) for the student. Windows are sized from the region itself, never a fixed pixel
count; the only guard is that a window is never magnified more than 2x. An image with no usable region gets
random windows instead (counted as `fallback_rate` in history.csv). The DINO loss then treats every region
as a sample: the student's partial view of the ship must match the teacher's fuller view of the same ship.

## Commands

PC (from the repository root, the suite's venv):
```
SARDet\.venv\Scripts\python.exe RegionSSL\make_regions.py --smoke          # region file for the 48 smoke images
SARDet\.venv\Scripts\python.exe RegionSSL\RegionDINO_scene_object\smoke_test.py
SARDet\.venv\Scripts\python.exe RegionSSL\RegionDINO_object\smoke_test.py
SARDet\.venv\Scripts\python.exe RegionSSL\make_regions.py                  # all 94,034 pool images, about 1 h on 8 cores
SARDet\.venv\Scripts\python.exe RegionSSL\make_regions.py --pool atrnet --smoke                        # arm I smoke region file
SARDet\.venv\Scripts\python.exe RegionSSL\RegionDINO_object\smoke_test.py --config config_ssl_atr.json  # arm I smoke (either method)
SARDet\.venv\Scripts\python.exe RegionSSL\chip_lab.py --n 300 --sizes 128,224                        # the rule on ATRNet chips
```

CRC, from the suite root `/ihome/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet` (the suite's job scripts take the
RegionSSL folders as their first argument; nothing else changes):
```
sbatch --job-name=regions ../RegionSSL/crc/make_regions.sbatch                       # once, CPU, ~1 h

# SSL, 100 epochs, one job chain per split (self-resuming), 4 configs x 3 splits = 12 chains:
for s in 1 2 3; do
  sbatch --job-name=rso_s$s  crc/ssl.sbatch ../RegionSSL/RegionDINO_scene_object $s
  sbatch --job-name=rsor_s$s crc/ssl.sbatch ../RegionSSL/RegionDINO_scene_object $s config_ssl_random.json
  sbatch --job-name=ro_s$s   crc/ssl.sbatch ../RegionSSL/RegionDINO_object $s
  sbatch --job-name=ror_s$s  crc/ssl.sbatch ../RegionSSL/RegionDINO_object $s config_ssl_random.json
done

# Arm I (in-domain): the region file for the ATRNet pool (CPU, ~20 min), then 5 configs x 3 splits = 15 chains:
sbatch --job-name=regions_atr ../RegionSSL/crc/make_regions_atr.sbatch
for s in 1 2 3; do
  sbatch --job-name=adino_s$s ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/DINO_reference $s config_ssl_atr.json
  sbatch --job-name=aro_s$s   ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_object $s config_ssl_atr.json
  sbatch --job-name=aror_s$s  ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_object $s config_ssl_atr_random.json
  sbatch --job-name=arso_s$s  ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_scene_object $s config_ssl_atr.json
  sbatch --job-name=arsor_s$s ../RegionSSL/crc/ssl_atr.sbatch ../RegionSSL/RegionDINO_scene_object $s config_ssl_atr_random.json
done
# their ATRNet fine-tunes: SARDet_to_ATR/crc.sbatch with methods atr_dino atr_regiondino_o atr_regiondino_o_random atr_regiondino_so atr_regiondino_so_random

# ATRNet-STAR fine-tunes (1 / 3 / 10 %), from the repository root, after the SSL chains finish:
cd .. && for s in 1 2 3; do for m in regiondino_so regiondino_so_random regiondino_o regiondino_o_random; do
  sbatch --job-name=xa_${m}_s$s SARDet_to_ATR/crc.sbatch $m $s; done; done

# SARDet detection fine-tunes (optional second table), from the suite root; the 4th argument picks the SSL run:
sbatch --job-name=rso_ft_s1 crc/finetune.sbatch ../RegionSSL/RegionDINO_scene_object 1 auto regiondino_so_ssl
sbatch --job-name=rsor_ft_s1 crc/finetune.sbatch ../RegionSSL/RegionDINO_scene_object 1 auto regiondino_so_random_ssl
```
Results back: commit the small files of each run folder (config.json, run_info.json, manifests.json,
history.csv, charts, eval_*; never `*.pth`/`*.pt`), push, pull on the PC, then
`python SARDet_to_ATR/compare.py` for the ATRNet table.

## Rows of the first comparison (ATRNet-STAR test accuracy, 1 / 3 / 10 % labels, 3 splits)

| row | SSL stage | status |
|---|---|---|
| A0 | none (ImageNet) | done |
| XA DINO / LeJEPA / SparK | plain SSL on SARDet scenes | CRC jobs submitted 2026-11-17 |
| XA RegionDINO scene+object | this folder | SSL chains running on CRC (2026-11-18) |
| XA RegionDINO object-only | this folder | done: 27.2 / 49.0 / 73.7 (+9.4 / +7.1 / +1.4 vs XA DINO) |
| the two random-window controls | this folder | object-only control done; scene+object control running |
| XI DINO (suite recipe) / RegionDINO scene+object / object-only / controls, SSL on the ATRNet pool | arm I, this folder | built 2026-11-18, smoke-tested on the PC, not run |
