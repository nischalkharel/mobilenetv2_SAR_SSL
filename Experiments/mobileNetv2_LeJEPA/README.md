# A2: MobileNetV2 + LeJEPA on SAR (ATRNet-STAR SOC-40)

```
A0:  IMAGENET1K_V2 weights ------------------------------------------> fine-tune on labeled_set.csv -> validation / test
A1:  IMAGENET1K_V2 weights -> DINO SSL on ssl_training_data.csv ------> fine-tune (same) -> validation / test
A2:  IMAGENET1K_V2 weights -> LeJEPA SSL on ssl_training_data.csv ----> fine-tune (same) -> validation / test
```

This folder is self-contained: it does not import anything from `Experiments/Baseline`, `Experiments/mobileNetv2_DINO` or `lejepa-main` at run time. Status: **implemented and smoke-tested; pilot and full run pending** (see "Status" at the end).

## What LeJEPA does, in plain words

There is **one network**: the MobileNetV2 backbone plus a small MLP projector. There is no teacher copy, no moving average, no stop-gradient and no centering (all of which A1's DINO needs).

Each training image is cut into several random views. All views go through the network and give one projection each. The loss has two parts:
- **agreement**: every view's projection should be close to the mean projection of the image's views (mean squared distance). On its own this has a trivial solution: map every image to the same point (collapse).
- **SIGReg**: the projections of a batch should look like samples from a standard Gaussian, in every direction. It is measured by projecting onto 1,024 random directions and running a normality test (Epps-Pulley) on each. Collapsed or squashed projections score high, so this term prevents collapse.

`loss = (1 - lambda) x agreement + lambda x SIGReg`, with lambda = 0.05 (the paper's default). After SSL, the projector is thrown away and the backbone is fine-tuned exactly like A0 and A1.

## Sources

| What | Where |
|---|---|
| Paper | Balestriero and LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", arXiv 2511.08544 **v3** (14 Nov 2025). Read on arxiv.org; the PDF is not in `Papers/` |
| Code | Official repository rbalestr-lab/lejepa, downloaded as a ZIP (`lejepa-main`, no Git commit available). Untouched copies of the files we use or cite: `reference/lejepa-main_original/`, hashes in `reference/SOURCE.md` |
| Reused unchanged | `lejepa_lib/` = the authors' `univariate/base.py`, `univariate/epps_pulley.py`, `multivariate/slicing.py` (SIGReg). Only the small `__init__.py` files are new, because the original package also imports tests that need scipy |
| Licence | The LeJEPA code is **CC BY-NC 4.0 (non-commercial)**, unlike DINO's Apache 2.0. Fine for university research; check before any use connected to commercial work |

Every change and every choice is listed in `LEJEPA_CHANGES.md`.

## Commands (Git Bash, from the repository root)

In the Claude app's Terminal panel (PowerShell), activate with `.\.venv\Scripts\Activate.ps1` instead of `source`.

**Step 0. Activate the environment.**
```bash
source .venv/Scripts/activate
```

**Step 1. Smoke test** (about 3 minutes, 50 checks, not a result). Writes only to `outputs/smoke/`.
```bash
python Experiments/mobileNetv2_LeJEPA/smoke_test.py
```
The last line should be `SMOKE TEST PASSED (both parts)`.

**Step 2 (optional). Look at the views** (`outputs/view_check/view_preview.png`, `crop_coverage.json`).
```bash
python Experiments/mobileNetv2_LeJEPA/preview_views.py
```

**Step 3. Full SSL run** (about 14 hours for 100 epochs). After 25 steps it prints the measured speed, the time left and the peak GPU memory; stop it with Ctrl+C if those look wrong.
```bash
python Experiments/mobileNetv2_LeJEPA/train_ssl.py
```

**Step 4. Resume** after an interruption; it continues from the last completed epoch in the same folder:
```bash
python Experiments/mobileNetv2_LeJEPA/train_ssl.py --resume "Experiments/mobileNetv2_LeJEPA/outputs/ssl/<SSL run folder>"
```
Resume refuses to continue if any setting, the image list or the schedule differs. The first run of this folder builds its own pixel cache (about 4 minutes, 1.1 GB in `outputs/cache/`).

**Step 5. Fine-tuning** (same code and settings as A0 and A1), from the **final** backbone:
```bash
python Experiments/mobileNetv2_LeJEPA/train_supervised.py --backbone "Experiments/mobileNetv2_LeJEPA/outputs/ssl/<SSL run folder>/backbone_epoch100.pth"
```

**Step 6. Validation evaluation.**
```bash
python Experiments/mobileNetv2_LeJEPA/evaluate.py --run "Experiments/mobileNetv2_LeJEPA/outputs/finetune/<fine-tuning run folder>" --split validation
```

**Step 7. Test evaluation**, once, only when every setting is frozen. Never use test results to choose settings or epochs.
```bash
python Experiments/mobileNetv2_LeJEPA/evaluate.py --run "Experiments/mobileNetv2_LeJEPA/outputs/finetune/<fine-tuning run folder>"
```

## SSL recipe (`config_ssl.json`)

"Authors" means the LeJEPA README, MINIMAL.md, launch scripts or paper. There is **no official MobileNetV2 recipe**; the closest author evidence for small CNNs is `scripts/launch_inet10.py` (a sweep over timm models under 20 M parameters on ImageNet-10) and the paper's Section 5.1 note for non-ViT backbones.

| Setting | Value | Authors | Origin of our value |
|---|---|---|---|
| `init` | ImageNet IMAGENET1K_V2 | random (from scratch) | Protocol: A2 starts from the same file as A0 and A1 (hash-checked) |
| Views | **8 views, all 224 px, "global"** | paper default for ViT: 2 global 224 + local 96; README: 2 global + 6 local; for CNNs the paper says "set global_views = all_views"; CNN sweep: `n_views=8`, `multi_crop=false`, resolution 224 | Author CNN guidance. Not DINO's 2 + 8 multi-crop. Every view is at the downstream input size, 224 |
| Crop scale | 0.3 to 1.0 | README global views (0.3, 1.0) | Authors. Checked on SAR with the XML vehicle boxes (2,000 images, 16,000 crops): 93.3% of crops keep at least half the vehicle, 0.4% contain none |
| Flip | horizontal, p 0.5 | same | Authors; vertical flips and rotations not added (range direction, as in A0/A1) |
| Brightness / contrast | 0.4 / 0.4, p 0.8 | same | Authors |
| Saturation, hue, random grayscale | left out | 0.2, 0.1, p 0.2 | They do nothing to a gray image, so leaving them out changes nothing |
| Blur | p 0.5, sigma 0.1 to 2 px | p 0.5; MINIMAL sigma 0.1 to 2 | Authors' probability and sigma range; same blur operation (PIL) as A1 |
| Solarization | **off** | p 0.2 | SAR choice, same as A1: it inverts exactly the bright scattering points that identify vehicles |
| Gray to 3 channels, ImageNet mean/std | same as A0/A1 | same normalization | Protocol |
| Projector | MLP 1280 -> 2048 -> 2048 -> **512**, BatchNorm + ReLU after the hidden layers | MINIMAL: MLP(512 -> 2048 -> 2048 -> proj_dim) with BatchNorm; CNN sweep: `projector_dim=512` | Authors' architecture and the CNN sweep's output size. Input is MobileNetV2's 1280 pooled features (the backbone is not changed). The paper's ViT ablation favours smaller outputs (64); not tested here |
| Agreement target | mean of all views' projections | paper: mean of the global views; CNN: all views are global | Same thing here. No stop-gradient |
| `lamb` | 0.05 | paper's recommended default | Authors |
| SIGReg | Epps-Pulley, 17 points on [0, 3], 1,024 random directions, new directions every step | README quick start: 17 points, 1,024 slices; library default t_max 3; paper text suggests [-5, 5] | Authors' library defaults. The paper's own ablation found the integration range has little effect |
| Optimizer | AdamW, lr 5e-4, weight decay 5e-4 on all parameters, no gradient clipping | README: lr 5e-4 "good starting point", weight decay 5e-4 "for ResNets" | Authors' CNN values |
| Schedule | linear warm-up for 1 epoch from 0.01 x lr, then cosine to lr / 1000 | README: warm-up + cosine to lr/1000; MINIMAL: 1-epoch warm-up from 0.01 x lr | Authors |
| Precision | bf16 autocast for the network; loss in float32 | bf16 everywhere | bf16 as the authors; the loss is computed in float32 (cheap, avoids precision loss in SIGReg's averages) |
| **Batch** | **28 images (224 views per step)** | 256 to 512 (paper: competitive from 128; SIGReg's bias "not a concern even for minibatches as small as 16") | **Forced by the GPU.** See "Time and memory": at 32 the process plus other programs exceed the card's 12 GB and Windows slows everything down |
| **Epochs** | **100** | 100 to 800 depending on the experiment | **Same number of epochs as A1** (Nischal's decision). This is not equal compute: about 14 h here against A1's 7.6 h. A time-matched run (55 epochs) is a later, separate experiment |
| Seed | 0 | 0 | SSL seed, separate from the fine-tuning seed |
| Checkpoint used downstream | final epoch | n/a | Fixed in advance. Backbones are also exported every 5 epochs for the accuracy-versus-epoch analysis only |
| Online linear probe | **not used** | MINIMAL trains one with labels during SSL | It would put labels into the SSL stage |

## Differences from A1 (DINO)

| | A1 DINO | A2 LeJEPA |
|---|---|---|
| Networks | student + teacher (moving average), teacher exported | one network, exported |
| Collapse prevention | teacher centering + sharpening, EMA, stop-gradient | SIGReg term |
| Head | 3-layer MLP -> 256 -> normalized 4,096-way layer | 3-layer MLP -> 512 |
| Views per image | 2 x 224 (scale 0.4-1) + 8 x 96 (0.15-0.4) | 8 x 224 (0.3-1) |
| Batch | 64 images | 28 images |
| Optimizer | AdamW, peak lr 1.25e-4 (scaled), wd 0.04, 10-epoch warm-up, gradient clip 3 | AdamW, peak lr 5e-4, wd 5e-4, 1-epoch warm-up, no clipping |
| Precision | fp16 + loss scaling | bf16 |
| Epochs, steps | 100 epochs, 102,100 steps | 100 epochs, 233,500 steps |
| Images seen | 6.53 M | 6.54 M |
| 224-px views through the network | 13.1 M (student) + 13.1 M (teacher, no backward) + 52.3 M at 96 px | 52.3 M, all with backward |
| SSL time on this PC | 458 min | about 860 min (estimated from the pilot) |

**Equal epochs do not mean equal compute**: one LeJEPA epoch costs about 1.8 times a DINO epoch here (8 full-size views with backpropagation, against 2 full-size and 8 small ones). The first A2 run uses the same number of epochs and images seen as A1 (Nischal's decision, to understand accuracy and metrics first), so it gets about 1.9 times A1's GPU time; report that next to any accuracy difference. A time-matched run (55 epochs) and a run with DINO's crops are separate, later experiments (`TODO.md`).

## What to watch during SSL

`history.csv` and `charts/`:
- `ssl_agreement.pdf`: falls as views of one image get closer. Near 0 together with a high SIGReg would mean collapse.
- `ssl_sigreg.pdf`: about 1 is what a true Gaussian gives at this batch size; it starts higher and should come down.
- `ssl_projector_stats.pdf`: the projections' standard deviation (target 1) and mean (target 0).
- `ssl_feature_similarity.pdf`: the same backbone-feature collapse check as A1 (cosine between images, spread).

A falling loss does **not** show that the features are useful. Only the fine-tuning result on validation can show that. (The paper reports that LeJEPA's loss correlates with downstream accuracy; that is not assumed here.)

## Time and memory (measured on this PC, 2026-10-16)

| Item | Measured |
|---|---|
| GPU step, 8 views x 224 px, random pixels | batch 16: 0.12 s, 5.6 GB reserved; 24: 0.18 s, 8.2 GB; 28: 0.21 s, 9.7 GB; 32: 0.31 s at best, 10.8 to 10.9 GB |
| Why not 32 | 10.9 GB for this process, plus the CUDA context and other programs' GPU memory (about 0.6 GB on this PC), is more than the card's 12 GB. Windows then pages GPU memory into system RAM: 12.8 ms per image instead of 7.4. Batch 64 crawled for the same reason |
| CPU, making 8 views of one image | 7.0 ms (one process); 4 workers deliver a batch faster than the GPU uses it |
| First pilot (batch 32, float RGB views from the workers, 2 workers) | 0.56 s/step, 18.3 min per epoch: data transfer plus the memory problem above. Folder renamed `..._OLD-pilot-batch32`; not used |
| **Pilot, final settings** (epoch 1 of a 55-epoch schedule; folder renamed `..._OLD-pilot-55ep`, not used) | **0.210 s/step, 8.56 min per epoch, peak 8.70 GB allocated** |
| Planned full run | 100 epochs x about 8.6 min = **about 14.3 hours** (A1: 7.6 hours) |

## Status

Implemented, smoke-tested (50 checks) and piloted for one epoch. Full 100-epoch run started 2026-10-16. No A2 result exists. Pilot epoch 1: loss 0.152, agreement 0.063, SIGReg 1.85 (from about 5 early in the epoch), projection std 0.84 (from 0.33), backbone feature cosine 0.304 (A1 after epoch 1: 0.340), feature std 0.022. These only show that training is stable; they say nothing about downstream accuracy.
