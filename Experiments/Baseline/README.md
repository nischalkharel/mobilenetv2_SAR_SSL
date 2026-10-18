# A0 baseline: ImageNet MobileNetV2 fine-tuned on the SOC-40 labeled set

Supervised classification baseline with no SAR self-supervised learning (SSL).

- **Model:** TorchVision MobileNetV2. The backbone (`model.features`) starts from `MobileNet_V2_Weights.IMAGENET1K_V2`, read from the saved file `MobileNetV2/checkpoints/pretrained/mobilenet_v2_imagenet1k_v2.pth`. The smoke test checks that file is identical to TorchVision's weights. The ImageNet 1000-class layer is replaced by a new, randomly initialized 40-class layer. The whole network is trained (full fine-tuning).
- **Data:** training uses only `preprocessing_dataset/labeled_set.csv` (1,961 images, 3% of the SSL pool, 35 to 68 per class; simulates a small labeled budget). Validation uses `preprocessing_dataset/validation.csv` (800 images, 20 per class, each from a different pose); it is evaluated after every epoch, and the best epoch is chosen with it. `ssl_training_data.csv` and `excluded_near_duplicates.csv` are not used. The official SOC-40 `test/` folder is used only by `evaluate.py`, after training is finished.
- **Later SSL routes (A1, A2, ...):** run exactly the same `train.py` and `evaluate.py` with a config that sets `"init": {"type": "backbone_checkpoint", "backbone_checkpoint": "<file>"}`. The SSL run saves its file as `{"backbone_state_dict": model.features.state_dict()}`. Everything else stays the same, so the only difference between routes is where the backbone starts.

## How to run the baseline

Open Git Bash and run everything from the repository root (the `mobilenetv2_SAR_SSL` folder). Before the first run, `paths.py` must point to the dataset on this PC (see the main README, STEP 0).

**Step 0. Go to the repository and activate the environment**
```bash
cd <path to>/mobilenetv2_SAR_SSL
source .venv/Scripts/activate
```

**Step 1 (optional). Pipeline check, about 1 minute. Not a result.**
```bash
python Experiments/Baseline/smoke_test.py
```
All lines should say `[PASS]`. It creates a small folder ending in `_SMOKE` in `Baseline_Output/`; that folder is not a result and can be deleted.

**Step 2. Train.** A few minutes on the GPU.
```bash
python Experiments/Baseline/train.py
```
- Every run creates a new folder, `Experiments/Baseline/Baseline_Output/<date-time>_A0_imagenet_mobilenetv2_seed0/`. Earlier runs are never overwritten.
- When the run finishes, it prints the exact Step 3 command for that folder.
- The `eval_test/` folder does not exist yet at this point.

**Step 3. Evaluate on the official test set.** Run this only when training is final, and once per final run.
```bash
python Experiments/Baseline/evaluate.py --run "Experiments/Baseline/Baseline_Output/<run folder from step 2>"
```
- This creates `eval_test/` inside that run folder.
- Test results must never be used to change settings or pick checkpoints.

**Options**

| Option | What it does |
|---|---|
| `train.py --device cpu` | Runs on the CPU instead of the GPU (for the CPU timing runs). Also works with `evaluate.py` and `smoke_test.py`. |
| `train.py --config <file>` | Uses another config, for example a copy of `config.json` with a different seed or a DINO backbone. |
| `evaluate.py --split validation` | Re-scores the validation set into `eval_validation/`, to check that a saved checkpoint reloads correctly. It does not touch the test set. |
| `python Experiments/Baseline/plots.py --run <run folder>` | Redraws all PDF charts of a run from its saved CSV files, for example for a run trained before the charts existed. It reads saved files only, and does not run the model or the test set. |

## Image handling (`data.py`, shared by every route)

Each image on disk is a 128x128, single-channel, 8-bit (0 to 255) GeoTIFF. The values are non-linearly quantized ground-range amplitude and are mostly dark: on a 0 to 1 scale, the mean is about 0.117, the standard deviation about 0.147, and about 0.9% of pixels are saturated at 1.0.

| Step | Training | Validation / test |
|---|---|---|
| Random translation | pad 8 pixels (reflect) and crop back to 128, so the chip shifts by up to 8 pixels | none |
| Horizontal flip | probability 0.5 | none |
| Resize | 128 to 224, bilinear | same |
| Gray to 3 channels | the single channel is copied into R, G and B (the backbone expects 3 channels; this adds no color information) | same |
| Scaling | divide by 255, giving values in [0, 1] | same |
| Normalization | ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225), matching what the pretrained backbone was trained with | same |

Why these choices:
- **224 input size.** The ImageNet weights were trained at 224, and upsampling usually transfers better than using the native 128. Running at 128 would be about 3x cheaper per image. That is a reasonable later experiment, because it affects CPU latency.
- **Horizontal flip only.** The range direction runs along the image height, so a left-right flip keeps radar shadows on the correct side and acts like a mirrored viewing angle. Vertical flips and rotations would move the shadow to a physically wrong side, so they are not used. No color or brightness changes are used.
- **Small shifts.** Targets are already placed off-center in this dataset, so a shift of up to 8 pixels mimics that natural variation. A target very close to the edge can lose a few pixels.
- Evaluation preprocessing is deterministic. `smoke_test.py` checks this and saves `sample_inputs.png` (training inputs on top, validation inputs below).

## Configuration (`config.json`, all settings in one place)

| Setting | Value | Reason |
|---|---|---|
| `train.optimizer` | `adamw` (`sgd` is also supported) | Robust default for fine-tuning a whole pretrained network |
| `train.lr` | 0.0005 | Moderate for full fine-tuning; the new head and the backbone share one learning rate |
| `train.weight_decay` | 0.01 | Usual AdamW value |
| `train.scheduler` | `cosine` over all epochs, stepped once per epoch | The learning rate starts at `lr` and falls along a half cosine curve to almost 0 at the last epoch: big steps early to adapt the pretrained network quickly, small steps late to settle into a good solution without bouncing around it. No extra settings to tune |
| `train.epochs` | 30 | 1,961 images = 31 batches per epoch, so about 930 updates; validation picks the best epoch. Check `charts/accuracy.pdf`: if validation is still rising at epoch 30, more epochs may be needed (decide with validation only) |
| `train.batch_size` | 64 | Fits in CPU and GPU memory |
| `train.seed` | 0 | Controls head initialization, data order and augmentation |
| `train.device` | `cuda` | GPU for now. CPU numbers are also a project goal: use `--device cpu` (or set `"cpu"`) for those runs. `cuda` fails with a clear error instead of silently falling back to the CPU; `auto` picks the GPU when available |
| `train.log_every_batches` | 25 | Prints a progress line every 25 batches (an epoch has 31 training and 13 validation batches, so mostly the end-of-epoch line appears); 0 turns it off. Screen output only, nothing saved |
| `data.input_size` | 224 | See above |
| `data.num_workers` | 0 | On this PC, background data-loading processes ran out of Windows memory and hung. Images are cached in RAM, so loading is not the bottleneck |
| `data.cache_in_memory` | true | Reads each image file from disk only once |

These are starting values, not tuned values. Tune only with validation results, never with test results. Each run saves the settings it actually used to its own `config.json`.

## What you see on screen

`train.py` prints:
1. A header with the output folder, device, initialization, image counts and settings.
2. One line while the train and validation images are read into memory. This happens once per run.
3. For each epoch:
   - a progress line every `log_every_batches` batches (running training loss and accuracy, seconds so far);
   - validation progress lines;
   - an epoch summary (train and validation loss and accuracy, `*new best, saved best.pt` when validation improves);
   - the best result so far and an estimate of the time left.
4. At the end, the list of files written (including the `charts/` folder) and the exact `evaluate.py` command for this run.

`evaluate.py` prints the checkpoint used, the split and device, progress lines, the final accuracy, and the list of files written.

## What each run folder contains (`Baseline_Output/<time>_<run_name>_seed<seed>/`)

| File | Contents |
|---|---|
| `config.json` | Exact settings used (plus `"smoke": true/false`) |
| `class_mapping.csv` | Copy of the 40-class label mapping |
| `manifests.json` | SHA-256 of the CSV files and a hash of the images actually used, with counts |
| `run_info.json` | Status (running / completed / failed), timing, Python/torch versions, device, Git commit and whether there were uncommitted changes, best epoch, best validation accuracy |
| `history.csv` | Per epoch: learning rate, training loss and accuracy, validation loss and accuracy, seconds |
| `best.pt` | Model weights at the best validation epoch, plus that epoch number, validation accuracy, class names and config |
| `charts/` | Written by `train.py`, redrawn after every epoch (can be opened during training). One PDF per chart: `loss.pdf` (train/val loss), `accuracy.pdf` (train/val accuracy, best epoch starred), `overfitting_gap.pdf` (train minus val accuracy), `learning_rate.pdf` |
| `eval_test/` | Written by `evaluate.py`: `metrics.json` (overall accuracy, mean per-class accuracy, loss, checkpoint hash, test-manifest hash), `per_class_accuracy.csv`, `confusion_matrix.csv` (rows = true class, columns = predicted class), `predictions.csv` (one row per image), and `charts/` with one PDF per chart: `per_class_accuracy.pdf` (lowest first), `confusion_matrix.pdf` (row-normalized), `top_mistakes.pdf` (25 most frequent true -> predicted mistakes), `confidence.pdf` (confidence of correct vs wrong predictions), `calibration.pdf` (accuracy vs confidence) |

Run folders are never overwritten. Smoke runs end in `_SMOKE` and are not results.

## Files

| File | Responsibility |
|---|---|
| `config.json` | All settings |
| `data.py` | Reads the subset CSVs and the test folder, checks labels, loads images, and builds the preprocessing |
| `model.py` | Builds MobileNetV2 with an interchangeable backbone initialization and a new 40-class head; reloads fine-tuned checkpoints |
| `engine.py` | One training epoch; prediction over a dataset |
| `train.py` | Training run: leakage guards, run records, per-epoch validation, best-checkpoint saving |
| `evaluate.py` | Loads `best.pt` and writes test (or validation) metrics, per-class accuracy, the confusion matrix and predictions |
| `smoke_test.py` | Pipeline checks plus a tiny 2-epoch run |
| `utils.py` | Seeding, hashing, environment and Git information |
| `plots.py` | PDF charts, one per file: `<run>/charts/` (called by `train.py`) and `<run>/eval_<split>/charts/` (called by `evaluate.py`); can also be run on its own |

## Known limitations

- **No resume.** An interrupted run must be restarted, and it gets a new folder.
- **GPU vs CPU.** The `.venv` has the CUDA build `torch 2.10.0+cu130` / `torchvision 0.25.0+cu130` (RTX 4070 SUPER, driver 591.86). The pip wheel bundles the CUDA runtime and cuDNN, so no separate CUDA toolkit is needed. With the earlier 10% labeled set (6,120 images), a 30-epoch run took about 11 minutes on the GPU (about 21 s per epoch); the 3% set is faster. GPU and CPU results are not bit-identical, and GPU runs with the same seed can differ slightly.
- **"CUDA out of memory" with plenty of GPU memory free.** On this PC, other programs that use the GPU (browsers, Electron apps, VS Code) can fill the GPU's BAR1 window (check with `nvidia-smi -q -d MEMORY`, "BAR1 Memory Usage"). Then CUDA fails with "out of memory" or "CUBLAS_STATUS_ALLOC_FAILED" even though the GPU itself has free memory. Closing those programs (or restarting the PC) fixes it; `--device cpu` works meanwhile.
- **Optimistic SOC-40 test accuracy.** The official SOC-40 test set shares pose groups with train (see `preprocessing_dataset/SUBSETS.md`), so its accuracy is likely somewhat optimistic for every route.
