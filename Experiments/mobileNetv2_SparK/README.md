# A3: MobileNetV2 + SparK on SAR (ATRNet-STAR SOC-40)

```
A0:  IMAGENET1K_V2 weights ------------------------------------------> fine-tune on labeled_set.csv -> validation / test
A1:  IMAGENET1K_V2 weights -> DINO SSL on ssl_training_data.csv ------> fine-tune (same) -> validation / test
A2:  IMAGENET1K_V2 weights -> LeJEPA SSL on ssl_training_data.csv ----> fine-tune (same) -> validation / test
A3:  IMAGENET1K_V2 weights -> SparK SSL on ssl_training_data.csv -----> fine-tune (same) -> validation / test
```

This folder is self-contained: it does not import anything from `Experiments/Baseline`, the DINO or LeJEPA experiments, or `SparK-main` at run time. Status: **implemented and smoke-tested (68 checks); full run pending** (see "Status").

## What SparK does, in plain words

SparK is masked image modeling for CNNs ("BERT for convnets"). Each view is cut into a 7 x 7 grid of 32-pixel patches and 60% of the patches are hidden. MobileNetV2 (the encoder) sees only the visible patches. The authors' "sparse" layers keep every hidden position at exactly 0 after every layer, and BatchNorm computes its statistics from visible positions only, so the hidden area neither leaks in nor distorts the statistics.

The encoder's feature maps at 4 resolutions are then "densified": hidden positions are filled with a learned mask token and projected to the decoder's width. A UNet-style decoder rebuilds the image from coarse to fine. The loss is the mean squared error on the **hidden** patches only, with each patch's pixels normalized to mean 0 and std 1, so the model has to predict structure rather than brightness. After SSL the decoder, the densify layers and the mask tokens are thrown away; the encoder is fine-tuned exactly like A0, A1 and A2.

How it differs from A1 and A2: those compare several views of an image in an embedding space and need a mechanism against collapse (teacher and centering in DINO, SIGReg in LeJEPA). SparK predicts pixels from one view, so it cannot collapse that way.

## Sources

| What | Where |
|---|---|
| Paper | Tian et al., "Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling", ICLR 2023 (spotlight), arXiv 2301.03580 **v2**. Read on arxiv.org; the PDF is not in `Papers/` |
| Code | Official repository keyu-tian/SparK, downloaded as a ZIP (`SparK-main`). All 18 files we use or cite are byte-identical to commit **a63e386** (2024-01-23, the latest on `main`). Untouched copies: `reference/SparK-main_original/`, hashes in `reference/SOURCE.md` |
| Reused | `spark_lib/`: the authors' `encoder.py`, `spark.py`, `decoder.py` (only import lines changed: timm is not installed), `lamb.py`, `lr_control.py` (unchanged) |
| Licence | MIT (copy in `spark_lib/LICENSE`) |

Every change and every choice is listed in `SPARK_CHANGES.md`.

## Commands (Git Bash, from the repository root)

In the Claude app's Terminal panel (PowerShell), activate with `.\.venv\Scripts\Activate.ps1` instead of `source`.

**Step 0. Activate the environment.**
```bash
source .venv/Scripts/activate
```

**Step 1. Smoke test** (about 5 minutes on the PC, 68 checks, not a result). Writes only to `outputs/smoke/`.
```bash
python Experiments/mobileNetv2_SparK/smoke_test.py
```
The last line should be `SMOKE TEST PASSED (both parts)`.

**Step 2 (optional). Look at the views and masks** (`outputs/view_check/view_preview.png`, `crop_and_mask_coverage.json`).
```bash
python Experiments/mobileNetv2_SparK/preview_views.py
```

**Step 3. Full SSL run.** On Pitt CRC: `sbatch crc/spark.sbatch` (smoke test, SSL, then fine-tuning on the 3%, 1% and 10% labeled sets with validation and test scores). Directly:
```bash
python Experiments/mobileNetv2_SparK/train_ssl.py
```
After 25 steps it prints the measured speed, the time left and the peak GPU memory.

**Step 4. Resume** after an interruption; it continues from the last completed epoch in the same folder:
```bash
python Experiments/mobileNetv2_SparK/train_ssl.py --resume "Experiments/mobileNetv2_SparK/outputs/ssl/<SSL run folder>"
```
Resume refuses to continue if any setting, the image list or the schedule differs. The first run of this folder builds its own pixel cache (about 4 minutes, 1.1 GB in `outputs/cache/`).

**Step 5. Fine-tuning** (same code and settings as A0, A1 and A2), from the **final** backbone. Add `--train-csv labeled_set_1pct.csv` or `labeled_set_10pct.csv` for the other label budgets.
```bash
python Experiments/mobileNetv2_SparK/train_supervised.py --backbone "Experiments/mobileNetv2_SparK/outputs/ssl/<SSL run folder>/backbone_epoch100.pth"
```

**Step 6. Evaluation.** Validation is for every decision; test scores are recorded but never used to choose settings (`crc/README.md`).
```bash
python Experiments/mobileNetv2_SparK/evaluate.py --run "Experiments/mobileNetv2_SparK/outputs/finetune/<fine-tuning run folder>" --split validation
python Experiments/mobileNetv2_SparK/evaluate.py --run "Experiments/mobileNetv2_SparK/outputs/finetune/<fine-tuning run folder>" --split test
```

## SSL recipe (`config_ssl.json`)

"Authors" means the SparK paper, README or `pretrain/` code. There is **no official MobileNetV2 or SAR recipe**: the authors use the same settings for every ResNet and ConvNeXt.

| Setting | Value | Authors | Origin of our value |
|---|---|---|---|
| `init` | ImageNet IMAGENET1K_V2 encoder; decoder side random | from scratch (their README suggests pre-trained weights for small datasets) | Protocol: same file as A0, A1, A2 (hash-checked) |
| Encoder maps | `features[3]` 24@56, `[6]` 32@28, `[13]` 96@14, `[18]` 1280@7 | ResNet `layer1`-`layer4` (strides 4-32) | Same strides; last layer of each resolution; `[18]` so every exported layer is trained |
| View | 1 per image, RandomResizedCrop 224, scale 0.67-1.0, flip p 0.5 | same, bicubic | Authors; bilinear + antialias as in A1/A2 (matches the fine-tuning resize) |
| Mask | 7 x 7 patches of 32 px, ratio 0.6 (20 of 49 visible) | 32 px, 0.6 | Authors. A vehicle spans about 2.8 x 2.8 patches; a median 39% of it stays visible; completely hidden in 0.5% of views |
| Densify norm, mask tokens | sparse BatchNorm, 4 learned tokens | `bn` for ResNets | Set explicitly: the authors' code picks it only for names containing "resnet"/"convnext" |
| Decoder | LightDecoder width 768 (5 up-sampling blocks), 23.2 M parameters | 768 for every encoder | Authors (about 10 times the 2.2 M encoder; discarded after SSL) |
| Target | **the SAR channel**, normalized per patch; loss on hidden patches only | 3 RGB channels, same normalization and loss | SAR change: with 3 channels a median 41% of the target would be the constant ImageNet channel offsets (`SPARK_CHANGES.md`) |
| Optimizer | LAMB, betas (0.9, 0.95), global gradient norm clipped at 5 | same (ResNet beta2) | Authors |
| Learning rate | peak 5e-5 = 2e-4 x 64 / 256, warm-up 2.5 epochs from 0.005 x peak, cosine to 0.001 x peak | base 2e-4 x batch / 256, 40 of 1,600 epochs warm-up | Authors' rule; warm-up = the same 2.5% of training |
| Weight decay | 0.04 -> 0.2 (cosine); none on biases, BatchNorm, mask tokens | same | Authors |
| Precision | **full float32, TF32 off** | float32 (no autocast in `main.py`; the paper mentions V100s, which have no TF32) | Authors; measured necessity (next section) |
| Batch | 64 | 4,096 | **Same as A1 and A2** |
| Epochs | 100 (102,100 steps) | 1,600 | **Same as A1 and A2** (Nischal's decision) |
| Seed | 0 | n/a | SSL seed, separate from the fine-tuning seed |
| Checkpoint used downstream | final epoch | n/a | Fixed in advance; backbones also exported every 5 epochs for the accuracy-versus-epoch analysis only |

## Precision: why TF32 is switched off

PyTorch computes float32 convolutions in TF32 (10 mantissa bits) by default on A100 and RTX 30/40 GPUs. Gradient error against a float64 reference, measured on 8 SSL images at initialization:

| Precision | Whole gradient | Encoder conv weights (median) |
|---|---|---|
| CPU float32 | 3% | 3% |
| GPU float32, TF32 off (**used**) | 11% | 9% |
| GPU TF32 (PyTorch default) | 102% | 62% |
| GPU bf16 autocast | 129% | 145% |

With TF32 the encoder's gradient would be mostly rounding noise. A likely reason: ImageNet normalization maps the dark SAR background to about -2.1, and BatchNorm follows every conv, so many gradients are small remainders of large sums that cancel. **This may affect A1 (fp16) and A2 (bf16) too; it has not been measured for them yet** (`TODO.md`). The cost: without TF32, an A100 is not faster than the PC's GPU.

## Differences from A1 and A2

| | A1 DINO | A2 LeJEPA | A3 SparK |
|---|---|---|---|
| Learns by | matching a teacher's output across views | agreeing views + SIGReg | reconstructing hidden pixels |
| Networks during SSL | student + teacher (moving average) | one network + projector | encoder + decoder (23.2 M) |
| Views per image | 2 x 224 + 8 x 96 | 8 x 224 | 1 x 224, 60% hidden |
| Collapse prevention | centering, sharpening, EMA | SIGReg | none needed (pixel targets) |
| Optimizer | AdamW, peak lr 1.25e-4, 10-epoch warm-up, clip 3 | AdamW, peak lr 5e-4, 1-epoch warm-up | LAMB, peak lr 5e-5, 2.5-epoch warm-up, clip 5 |
| Precision | fp16 + loss scaling | bf16 | float32, TF32 off |
| Batch, epochs, steps | 64, 100, 102,100 | 64, 100, 102,100 | 64, 100, 102,100 |
| Exported | teacher backbone | backbone | encoder (= backbone) |

## What to watch during SSL

`history.csv` and `charts/`:
- `ssl_loss.pdf`: starts at about 1.0 (what predicting each patch's mean gives) and should fall. If it stays at 1.0, the decoder learns nothing.
- `reconstruction_epoch<NNN>.png` (every 5 epochs): hidden patches should start to show plausible structure (bright scatterers, edges).
- `ssl_gradient_norm.pdf`: the red line is the clipping threshold (5).
- `ssl_feature_similarity.pdf`: the same encoder collapse checks as A1/A2, over visible positions.

A falling loss does **not** show that the features are useful. Only the fine-tuning result on validation can show that.

## Time and memory

| Item | Measured |
|---|---|
| PC GPU (RTX 4070 SUPER), float32 TF32 off | batch 8: 13.1 ms/image, 3.1 GB; batch 16: 10.1 ms, 3.9 GB; batch 24: 9.3 ms, 5.1 GB (peak allocated) |
| Batch 64 | about 11 GB by extrapolation: fits a 40 GB A100 |
| A100, 100 epochs | not measured; roughly 12 to 17 h expected (float32 without TF32). `crc/spark.sbatch` allows 36 h |

## Status

Implemented and smoke-tested (68 checks, 2026-10-27, PC GPU), then reviewed line by line by an independent reviewer (no error in the training computation or the export; its robustness points are fixed). Problems found and fixed before any run: resume drew different masks than an uninterrupted run (an export consumed the global random generator); TF32 (see "Precision"); a stop right after a checkpoint could lose that epoch's history row and export (now rebuilt on resume). No A3 result exists yet.
