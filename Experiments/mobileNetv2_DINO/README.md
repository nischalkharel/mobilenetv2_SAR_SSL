# A1: ImageNet MobileNetV2 + DINO on unlabeled SAR + supervised fine-tuning

Route A1 of the study. It is compared with A0 (`Experiments/Baseline`):

```
A0:  IMAGENET1K_V2 weights ----------------------------------------> fine-tune on labeled_set.csv -> validation / test
A1:  IMAGENET1K_V2 weights -> DINO on ssl_training_data.csv (no labels) -> fine-tune on labeled_set.csv -> validation / test
                              (stage 1: train_ssl.py)                    (stage 2: train_supervised.py, same as A0)
```

Both routes start from the same ImageNet file, fine-tune the whole network on exactly the same 1,961 labeled images with exactly the same settings, select the epoch with validation (800 images), and are tested once on the official SOC-40 test set. The only difference is the extra DINO stage. That stage sees many more images and costs extra compute; that cost is recorded separately in the SSL run's `run_info.json` (`cost`) and copied into every fine-tuning run that uses its backbone.

This folder is self-contained: it imports nothing from `Experiments/Baseline` or from `dino-main`. External inputs (configured in `paths.py`): the ATRNet-STAR images, the subset CSVs in `preprocessing_dataset/`, and the saved ImageNet weights.

## Data flow

1. **Start.** Student = MobileNetV2 `features` loaded from `mobilenet_v2_imagenet1k_v2.pth` (the A0 file), then global average pooling, giving 1280 values per image. A new DINO head turns those into 4096 values (3-layer MLP 1280 -> 2048 -> 2048 -> 256, L2 normalization, weight-normalized linear layer 256 -> 4096). The teacher is an exact copy of the student. The ImageNet classifier is not used anywhere.
2. **DINO (stage 1, `train_ssl.py`).** Each unlabeled image gives 10 views: 2 global (224 px) and 8 local (96 px). Each step:
   - the teacher sees the 2 global views; the student sees all 10;
   - the loss makes the student's output for each view match the teacher's (centered, sharpened) output for the *other* global view(s);
   - only the student is trained by the optimizer. The teacher's weights become a slow moving average (EMA) of the student's.
3. **Export.** After the last epoch, the teacher's MobileNetV2 `features` (all weights and BatchNorm statistics) are saved as `backbone_teacher_epoch<N>.pth`. The DINO head is thrown away.
4. **Fine-tuning (stage 2, `train_supervised.py`).** The exported backbone is loaded into a fresh MobileNetV2, and a new Dropout 0.2 + Linear 1280 -> 40 head is attached. The whole network is then trained exactly like A0.
5. **Evaluation (`evaluate.py`).** Validation any time. The official test set only once, at the end.

Seeds: the SSL seed (`config_ssl.json`, `setup.seed`) and the downstream seed (`config_finetune.json`, `train.seed`) are separate settings, and they run in separate processes. The downstream seed is also reset right before the model is built. So the downstream head initialization, data order and augmentation are the same as A0's for the same seed, whatever the SSL run did.

## Commands (Git Bash, from the repository root)

**Step 0. Activate the environment.** Windows needs a page file for the SSL run; see "Known limitations".
```bash
cd <path to>/mobilenetv2_SAR_SSL
source .venv/Scripts/activate
nvidia-smi                     # other programs should use little GPU memory
```

**Step 1. Smoke test: about 2 minutes, 50 checks, not a result.** It writes only to `outputs/smoke/`, which can be deleted.
```bash
python Experiments/mobileNetv2_DINO/smoke_test.py               # on the GPU
python Experiments/mobileNetv2_DINO/smoke_test.py --device cpu  # same checks on the CPU (resume must match exactly)
```
Every line should say `[PASS]` and the last line `SMOKE TEST PASSED (both parts)`.

**Step 2 (optional). Look at the views.** This writes `view_preview.png` and `crop_coverage.json` to `outputs/view_check/`.
```bash
python Experiments/mobileNetv2_DINO/preview_views.py
```

**Step 3. DINO SSL training.** This is the long stage, about 100 epochs (see "Time and memory").
```bash
python Experiments/mobileNetv2_DINO/train_ssl.py
```
- The first run decodes all 65,387 images into `outputs/cache/` (about 4 minutes, 1.1 GB). Later runs reuse that cache.
- It prints the settings, the measured seconds per step, the estimated total time and the peak GPU memory.
- A checkpoint is saved after every epoch. At the end it exports the teacher backbone and prints the Step 5 command.

Stop early on purpose, for example to run a pilot that is the first part of the same schedule:
```bash
python Experiments/mobileNetv2_DINO/train_ssl.py --stop-after-epoch 5
```

**Step 4. Resume** after an interruption or a deliberate stop. It continues from the last completed epoch in the same folder.
```bash
python Experiments/mobileNetv2_DINO/train_ssl.py --resume "Experiments/mobileNetv2_DINO/outputs/ssl/<SSL run folder>"
```
Resume uses the run's own `config.json` and refuses to continue if the settings, the image list or the schedules differ. To check a config file against the run, add `--config <file>`; any difference is an error.

**Export again (normally not needed;** Step 3 exports automatically at the end):
```bash
python Experiments/mobileNetv2_DINO/export_backbone.py --run "Experiments/mobileNetv2_DINO/outputs/ssl/<SSL run folder>"
```

**Step 5. Supervised fine-tuning (A1).** A few minutes on the GPU.
```bash
python Experiments/mobileNetv2_DINO/train_supervised.py --backbone "Experiments/mobileNetv2_DINO/outputs/ssl/<SSL run folder>/backbone_teacher_epoch100.pth"
```
Parity check: this runs A0 from this folder's copy of the code.
```bash
python Experiments/mobileNetv2_DINO/train_supervised.py --init imagenet
```

**Step 6. Validation evaluation.** It re-scores the selected checkpoint on the 800 validation images, using the run's own manifest copies.
```bash
python Experiments/mobileNetv2_DINO/evaluate.py --run "Experiments/mobileNetv2_DINO/outputs/finetune/<fine-tuning run folder>" --split validation
```

**Step 7. Final test evaluation.** Run it once, only when every setting is frozen. Never use test results to choose settings, epochs, or teacher versus student.
```bash
python Experiments/mobileNetv2_DINO/evaluate.py --run "Experiments/mobileNetv2_DINO/outputs/finetune/<fine-tuning run folder>"
```

`--device cpu` works for `train_ssl.py` (new runs only), `train_supervised.py`, `evaluate.py` and `smoke_test.py`.

## SSL recipe (`config_ssl.json`)

Key names follow the original `main_dino.py` arguments. "Original" means the original code's default.

| Setting | Value | Original | Origin of the value |
|---|---|---|---|
| `init` | `imagenet` | random | Protocol: A1 starts from the same ImageNet weights as A0 |
| `out_dim` | 4096 | 65536 | Experimental choice. 65536 was chosen for ImageNet (1.28 M images); our pool is 65,387 images. Its last layer would be 16.8 M parameters, 7 times the whole backbone. It is not tied to the 40 classes. |
| head | 3 layers, hidden 2048, bottleneck 256, no BatchNorm, normalized last layer | same | Original |
| `momentum_teacher` | 0.9995, rising to 1 by a cosine schedule | 0.996 | The original's own advice for smaller batches ("use 0.9995 with batch size of 256"). Ours is 64 |
| `teacher_temp`, `warmup_teacher_temp`, `warmup_teacher_temp_epochs` | 0.04, 0.04, 0 | same | Original defaults |
| `student_temp`, `center_momentum` | 0.1, 0.9 | same | Original |
| `optimizer` | AdamW, with parameter groups: no weight decay on biases and BatchNorm weights | same | Original |
| `lr` | 0.0005 per 256 images | same | **This is a reference rate.** The rate actually used is lr x batch / 256 = **0.000125** at batch 64, as in the original. It is printed and saved as `effective_peak_lr` |
| `warmup_epochs`, `min_lr` | 10, 1e-6 | same | Original. Linear warm-up from 0, then cosine down to `min_lr` |
| `weight_decay`, `weight_decay_end` | 0.04, 0.04 (constant) | 0.04 rising to 0.4 | Experimental choice. The original notes that the increase "improves performance for ViTs"; it was not carried over to a CNN |
| `clip_grad` | 3.0 | same | Original |
| `freeze_last_layer` | 1 epoch | same | Original: the last head layer is not updated during epoch 1 |
| `use_fp16` | true (GPU only) | same | Original (mixed precision) |
| `batch_size_per_gpu` | 64, one GPU | 64 per GPU on 8 or more GPUs | Practical. It fits on the RTX 4070 SUPER. Smaller total batch than the original |
| `epochs` | 100 | 100 | Original default. The checkpoint used downstream is the **final** epoch, chosen in advance |
| `seed` | 0 | 0 | SSL seed; separate from the downstream seed |
| `num_workers` | 2 | 10 | This PC: 4 or more worker processes crashed for lack of memory (see limitations) |

**Views (`views` section).** Each image is 128 x 128 pixels, single channel; the vehicle box is a median 44 x 45 pixels, 12% of the image.

| Step | Value | Original | Why |
|---|---|---|---|
| Global crops | 2 x 224 px, area 40 to 100% of the chip | same | Original. 97% of global crops keep at least half of the vehicle (measured with the XML boxes) |
| Local crops | 8 x 96 px, area **15 to 40%** | 5 to 40% | SAR choice; see "Local crop scale" below. **Crops do not always contain the vehicle** (about 7% contain none) |
| Resize | bilinear, antialias | bicubic | Same as the baseline's resize |
| Horizontal flip | p 0.5 | same | Same reason as the baseline: range runs along the image height, so a left-right flip keeps shadows physically valid. No vertical flips or rotations |
| Brightness / contrast | 0.4 / 0.4, applied with p 0.8 | same | Kept as gain variation, which is plausible for SAR amplitude |
| Saturation, hue, random grayscale | removed | 0.2, 0.1, p 0.2 | They do nothing on a gray image |
| Gaussian blur | p 1.0 / 0.1 / 0.5 (global 1, global 2, local), radius 0.1 to 2 px at the crop's output size | same | Kept. Resembles a resolution or speckle-smoothing change. At 224 px a 2 px radius is about 1.1 px at native resolution |
| Solarization | **off** (p 0) | p 0.2 on the second global view | SAR choice. It inverts every pixel above 128, i.e. exactly the bright scattering points that identify vehicles in these mostly dark images |
| Gray to 3 channels, [0, 1], ImageNet mean/std | same as A0 | same normalization | Matches the baseline and the pretrained first layer |

**Local crop scale (15 to 40%).** The upper bound 0.4 is DINO's own boundary: global crops start at 40% of the image, so local crops are always the smaller views. Only the lower bound changed, for a geometric reason. DINO's 5% was set for 224 px ImageNet photos. On a 128 px chip, 5% is a crop of about 29 x 29 px, smaller than the median 44 x 45 px vehicle, and it is then upsampled 3.3x to 96 px. At 15%, the smallest crop is about 50 x 50 px: about the size of one vehicle, upsampled 1.9x. Measured with the XML vehicle boxes on 2,000 SSL-pool images, 8 crops each (`preview_views.py`):

| Lower bound | Crops with no vehicle | Keep < 25% of vehicle | Keep >= 50% of vehicle |
|---|---|---|---|
| 0.05 (original) | 14.3% | 35.4% | 41.3% |
| 0.10 | 10.0% | 28.1% | 47.1% |
| **0.15 (used)** | **6.9%** | **21.5%** | **54.3%** |
| 0.20 | 4.9% | 16.3% | 60.3% |
| 0.25 | 3.1% | 11.7% | 66.6% |

The curve is smooth, so 0.15 is a reasoned setting (smallest crop about one vehicle), not a measured optimum. Raising it further makes local crops more like global crops, which weakens DINO's local-to-global task. No downstream or validation result was used to choose it. The boxes are annotations from the SSL pool (no class labels, no validation or test images); they were used only for this design check, never in training.

`view_preview.png` in each SSL run shows original chips beside the exact views used. The table above summarizes what distinguishes this adaptation from original DINO; `DINO_CHANGES.md` has every line.

## What to watch during SSL

`history.csv` and `charts/`:
- `ssl_loss.pdf`: the DINO loss.
- `ssl_teacher_entropy.pdf`, which separates two kinds of collapse:
  - per-image entropy of the teacher's output: near ln(4096) = 8.32 means flat outputs (uniform collapse);
  - entropy of the batch average: near 0 means every image is sent to the same output (one-hot collapse);
  - in healthy training, the per-image entropy falls while the batch-average entropy stays high.
- `ssl_feature_similarity.pdf`: the mean cosine similarity between different images' 1280-value teacher features (1 = all images look identical) and their spread.
- `ssl_schedules.pdf`.

A finite or falling DINO loss does **not** show that the features are useful. Only the fine-tuning result on validation can show that.

## Time and memory (measured on this PC, 2026-10-13)

| Item | Measured |
|---|---|
| Pixel cache, first time | 223 s for 65,387 images (1.07 GB file in `outputs/cache/`); later runs reuse it in seconds |
| Making the 10 views of one image (CPU, one process) | 8.2 ms, so about 9 minutes per epoch with no workers; with 2 workers roughly half that. This, not the GPU, is likely to set the epoch time on this PC |
| Peak GPU memory of one training step (2026-10-14, random pixels, real shapes) | batch 16: 2.4 GiB; 32: 4.7 GiB; 48: 6.9 GiB; **64: 9.0 GiB** (of 12 GiB). It fits the card |
| Full-batch GPU step time | **Not measured yet.** Pilot attempts failed with "CUDA out of memory" while GPU memory was free: on Windows, GPU allocations also use system commit memory, and with no page file it ran out (measured: free commit fell to 0.0 GiB at the failure). See Known limitations |

Once a page file is enabled, measure with a pilot: the first epoch of the real schedule, resumable later.
```bash
python Experiments/mobileNetv2_DINO/train_ssl.py --stop-after-epoch 1
```
It prints the measured seconds per step, an estimate for all 100 epochs and the peak GPU memory after 25 steps. The SSL run also records its total cost in `run_info.json` (`cost`: minutes, images seen, views seen, optimizer steps, peak GPU memory). If the estimate is too long, decide on fewer epochs **before** the run; the checkpoint used is the final one of the planned schedule.

## Outputs

| Folder | Contents |
|---|---|
| `outputs/ssl/<time>_A1_dino_ssl_mobilenetv2_seed0/` | `config.json`; `manifests.json` with SHA-256 of the CSVs used, a manifest hash of the image list (sorted paths) and the split audit; `manifests/` with copies of `ssl_training_data.csv` and `class_mapping.csv`; `run_info.json` with status, environment, Git state, ImageNet file hash, pixel-cache hash and cost; `history.csv`; `charts/`; `view_preview.png`; `checkpoint.pth`, the full state for resume (student, teacher, optimizer, center, scaler, random states, epoch and step); `backbone_teacher_epoch<N>.pth` |
| `outputs/finetune/<time>_A1_dino_mobilenetv2_seed0/` | Same as an A0 run folder (`config.json`, `history.csv`, `best.pt`, `charts/`, `eval_validation/`, `eval_test/`) plus `manifests/`. Its `run_info.json` has the backbone file's SHA-256 and the full SSL record: run folder, checkpoint hash, SSL config, SSL seed, SSL manifest, SSL cost |
| `outputs/smoke/` | Smoke runs only (names end in `_SMOKE`). Not results |
| `outputs/cache/` | Decoded pixels for SSL. Safe to delete; rebuilt when needed |

Two hashes, not to be confused:
- the **manifest hash** identifies which files are in a list;
- the **pixel-cache hash** (`sha256_of_pixels`) identifies the decoded image contents.

## Files

| File | Responsibility |
|---|---|
| `paths.py` | All paths for this experiment (edit `ATRNET_GROUND_RANGE` per PC; it is separate from the repository's `paths.py` so this folder stands alone) |
| `config_ssl.json` / `config_finetune.json` | SSL settings / fine-tuning settings (fine-tuning values identical to A0) |
| `dino.py` | Copied DINO components (marked blocks), the MobileNetV2 feature adapter, and the collapse diagnostics |
| `ssl_data.py` | SSL image list (no labels), pixel cache, split audit, config flattening |
| `train_ssl.py` | DINO training, checkpoint and resume, export at the end |
| `export_backbone.py` | Teacher (default) or student backbone to the downstream checkpoint contract |
| `preview_views.py` | View preview and vehicle-coverage measurement of the crop settings |
| `train_supervised.py`, `evaluate.py`, `data.py`, `model.py`, `engine.py`, `plots.py`, `utils.py` | Supervised stage, copied from the baseline (differences listed in `DINO_CHANGES.md`) |
| `smoke_test.py` | All pipeline checks (two parts, each in its own process) |
| `tools/make_change_record.py` | Regenerates `diffs/` and the generated table in `DINO_CHANGES.md` |
| `DINO_CHANGES.md`, `diffs/` | Exact record of every change against the original DINO code and the baseline |
| `reference/` | Untouched copies of the original DINO files (with license) and of the baseline files, with hashes (`reference/SOURCE.md`) |

## Known limitations

- **Single run, single seed.** One SSL seed and one downstream seed. Differences between A0 and A1 smaller than seed-to-seed variation will need repeated runs before any claim.
- **Unequal compute by design.** A1 sees about 6.5 million extra images (65,387 x 100 epochs, 10 views each) during SSL. That is the question being asked (does extra SAR SSL help?), not a matched-compute comparison.
- **Recipe not tuned.** The SSL settings are a documented starting point. If they are tuned, tune them only with validation results, and count those validation labels as part of the supervision budget.
- **Resume.** It works at epoch boundaries only. A run stopped in the middle of an epoch restarts that epoch. On the GPU, a resumed run matched the uninterrupted one exactly in the smoke test, but cuDNN does not guarantee this; on the CPU it is exact.
- **This PC's memory.** Windows has no page file, so the commit limit equals RAM (31.8 GiB), and other programs use about 20 GiB of it. On Windows, GPU memory allocations also count against commit, so a batch-64 DINO step (9 GiB of GPU memory) can exhaust the roughly 11 GiB of free commit. Measured on 2026-10-14: the failure happens exactly when free commit reaches 0.0 GiB, while the GPU still reports free memory. (An earlier note blamed the GPU's full BAR1 window; that was wrong: 11.25 GiB could be allocated on the GPU with BAR1 full.) **Fix: enable a system-managed page file and restart.** Observed effects:
  - 4 or more data-loading workers crashed;
  - one long process running both SSL and fine-tuning failed with out-of-memory or access-violation errors;
  - "CUDA out of memory" appeared with GPU memory free.

  That is why `num_workers` is 2 and the smoke test runs its two parts as separate processes. Once, the smoke test's fine-tuning part crashed with exit code 3221225477 (a Windows access violation) right after the SSL part; it passed when rerun (`smoke_test.py --part downstream`). Adding a page file should also allow more workers and a faster SSL epoch.
- **Mixed precision may skip the first optimizer step(s).** The gradient scaler starts at a large scale and skips a step whose gradients overflow, then lowers the scale. The original DINO behaves the same way.
- **Optimistic SOC-40 test accuracy.** Official train and test share poses (see `preprocessing_dataset/SUBSETS.md`). This affects A0 and A1 alike.
