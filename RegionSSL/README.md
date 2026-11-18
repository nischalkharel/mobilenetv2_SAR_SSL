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
```

CRC, from the suite root `/ix/ageorge/nik96/mobilenetv2_SAR_SSL/SARDet` (the suite's job scripts take the
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
| XA RegionDINO scene+object | this folder | smoke-tested, not run |
| XA RegionDINO object-only | this folder | smoke-tested, not run |
| the two random-window controls | this folder | smoke-tested, not run |
