# SARDet_chips: chip classification on SARDet-100K

SSL on the SARDet-100K scenes (the backbones that already exist), then supervised fine-tuning and testing on
object chips cut from the same dataset's detection boxes (six classes). The protocol and the reasons behind it are
in [DESIGN.md](DESIGN.md). Study S2 (SSL on blind tiles) is parked in the tracker's idea box.

## Files

| file | job |
|---|---|
| `chip_paths.py` | where chips live (`SARDET_CHIPS_ROOT`, env var), the chip rule constants, and the suite's dataset paths re-exported |
| `make_chips.py` | cuts every chip once into one array file + manifest + summary + preview grids (30 s on the PC) |
| `chip_data.py` | which chips form the labeled set / selection set / test / test_clean, and the in-memory dataset |
| `config_finetune.json` | the ATRNet fine-tune recipe (30 epochs, AdamW 5e-4, cosine, batch 64, input 224) |
| `finetune_chips.py` | one fine-tune: method, budget, split -> run folder with records, best.pt, eval_test/, eval_test_clean/ |
| `compare.py` | tables over all completed runs (mean per-class and overall accuracy, paired differences, per class, per source) |
| `crc/env.sh`, `crc/make_chips.sbatch`, `crc/finetune.sbatch` | CRC: chip cutting as a CPU job, fine-tunes as GPU jobs with the chip array staged to node disk |

## Run

PC (data preparation and smoke checks only; training happens on CRC):

```
.venv\Scripts\python.exe SARDet_chips\make_chips.py
.venv\Scripts\python.exe SARDet_chips\finetune_chips.py --method imagenet --budget 1pct --split 1 --smoke --device cpu
.venv\Scripts\python.exe SARDet_chips\compare.py
```

CRC, from the repository root:

```
sbatch --job-name=make_chips SARDet_chips/crc/make_chips.sbatch          # once
for s in 1 2 3; do for m in imagenet dino lejepa spark regiondino_so regiondino_so_random regiondino_o regiondino_o_random; do
  sbatch --job-name=ch_${m}_s$s SARDet_chips/crc/finetune.sbatch $m $s; done; done     # 24 jobs, 3 budgets each
```

## What a run records

`config.json` (recipe, backbone, chip rule), `manifests.json` (chip array and manifest hashes, the image lists used,
chip counts per class, content hashes of every set), `run_info.json` (status, timing, commit, backbone SHA-256,
selected epoch, test numbers), `history.csv`, `best.pt`, `eval_test/` and `eval_test_clean/` (metrics.json with
overall, mean per-class, per-class and per-source accuracy; confusion matrix; per-chip predictions).

Chip sets: train = chips of the suite's `split<k>/labeled_<budget>.csv` images; selection = chips of `val_select.csv`
(2,000 val images); test = every chip of an official test image; test_clean = test minus images that share content
with train minus chips that hold another class inside the window. Headline metric: mean per-class accuracy on test_clean.
