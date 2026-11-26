# SARDet_to_ATR: does SSL on bigger, richer SAR images make a better backbone for ATRNet-STAR?

Question (from the 2026-11-13 meeting, Brittany's intent for SARDet-100K): the ATRNet-STAR chips are 128 px
and get stretched to 224 px and cropped during SSL. If the self-supervised stage instead sees SARDet-100K's
large, varied scenes (256 to 1000 px, nine sensors), is the backbone better or worse for the ATRNet-STAR
classification task? Everything downstream stays exactly as in the A0 to A3 and P4 tables: same labeled sets
(1 / 3 / 10 % of splits 1 to 3), same fine-tune config, same validation rule, same official test set. Only the
images the SSL stage saw change.

## Arms

| Arm | SSL stage | Status |
|---|---|---|
| A0 to A3, P4 | ATRNet chips (existing rows, nothing re-run) | done |
| **A** | SARDet-100K scenes, the 9 backbones already trained for the detection study (DINO / LeJEPA / SparK, 3 seeds), fine-tuned on ATRNet | this folder, ready to run |
| C | one mixed pool: ATRNet chips + SARDet scenes, new SSL runs | next, after A |
| **F** | FAIR-CSAR tiles (500 epochs), the 15 backbones from FAIR_chips (DINO, RegionDINO object-only, random control, teacher regions, teacher-size control; 3 splits), fine-tuned on ATRNet (XF rows) | done 2026-11-26 (c13): plain DINO +1.8 / +4.1 / +2.4 over no SSL, object-only 26.8 / 49.9 / 74.7, placement +2.8 / +1.2 / +0.7 (9 of 9), teacher = rule |
| B (object-scale-matched crops) | skipped by decision 2026-11-17 | not planned |

Reading rule: if A beats A1/A2/A3 at the low budgets, bigger and more varied SSL data helps the compact model;
if it loses, the chip-level SSL was not limited by image quality (object scale differs between the two
datasets, so a loss can also mean a scale mismatch; the mixed pool in C separates the two).

## Pairing and leakage

ATRNet split s uses the SARDet SSL run of split s (SSL seed s - 1), so the three pairs vary the SSL seed and the
labeled draw together, as the A1/A2/A3 rows did. The SARDet pool contains no ATRNet image; the ATRNet test set
is never seen by any SSL stage. The SARDet backbones were exported with torchvision `features.*` keys, the
same contract the ATRNet fine-tune code loads strictly.

## Files

- `config_finetune.json`: the A1 fine-tune config with the run name `XA_<method>_sardetssl_mobilenetv2`.
- `finetune_atr.py`: wrapper around `Experiments/mobileNetv2_DINO/train_supervised.py` (unchanged) writing to `SARDet_to_ATR/outputs/`.
- `find_backbone.py`: the SARDet backbone for a method and split, with the pairing rule above.
- `crc.sbatch <method> <split>`: fine-tunes 1 / 3 / 10 % and scores validation + test; skips budgets already scored.
- `compare.py`: table of all rows (A0, A1 to A3, P4, XA, later XC) with per-split paired differences.
- `response_maps.py` (+ `crc_response_maps.sbatch`): frozen-backbone response maps on the official ATRNet test chips for
  every SARDet-trained backbone, before any fine-tuning: share of the map's mass inside the vehicle box, concentration,
  peak-in-box, paired per chip against the ImageNet start and against XA DINO. Shows whether a backbone (RegionDINO in
  particular) responds on the vehicle rather than the background. Outputs: `outputs/compare/response_maps.{md,png}` + CSVs.

## Commands

PC smoke (needs a SARDet backbone file; the smoke export from the detection suite will do):
```
.\.venv\Scripts\python.exe SARDet_to_ATR\finetune_atr.py --method dino --backbone SARDet\DINO_SARDet\outputs\smoke\<run>\backbone_epoch2.pth --smoke
```
CRC, from the repository root (9 jobs, about 1.5 h each):
```
for m in dino lejepa spark; do for s in 1 2 3; do sbatch --job-name=xa_${m}_s$s SARDet_to_ATR/crc.sbatch $m $s; done; done
```
Results back: commit `SARDet_to_ATR/outputs/finetune/*/{config.json,run_info.json,history.csv,manifests.json,eval_*}`
(no `best.pt`), push, pull, `python SARDet_to_ATR/compare.py`.
