# DINO_CHANGES: what was copied, what was changed, and why

This file is the change record for route A1 (ImageNet MobileNetV2, then DINO on unlabeled SAR, then supervised fine-tuning). It has five sections:

1. Source and identity of the original code.
2. Plain-English explanation of each substantive adaptation.
3. Components of the original that were not copied.
4. Differences between the copied downstream code and the baseline, and the parity check.
5. A generated, line-by-line table of every changed line in every copied DINO block, plus the diff summary of the downstream files. The exact diffs are in `diffs/`.

Regenerate section 5 and `diffs/` after any edit to a copied block:
```bash
python Experiments/mobileNetv2_DINO/tools/make_change_record.py
```
The tool refuses to write anything if a block marked "unchanged" differs from the original, if any changed hunk has no written explanation, or if a reference copy no longer matches its recorded hash.

## 1. Source

- Original code: the local folder `dino-main` (facebookresearch/dino, Apache License 2.0, Copyright (c) Facebook, Inc. and its affiliates). It is a downloaded ZIP with no `.git` folder, so **no commit hash is available** and none is claimed. Untouched copies of `main_dino.py`, `utils.py`, `vision_transformer.py`, `README.md` and `LICENSE`, with their SHA-256 hashes, are in `reference/dino-main_original/` and `reference/SOURCE.md`.
- Every copied block in `dino.py` and `train_ssl.py` sits between `# >>> DINO extract <ID>: <file> L<first>-L<last> (<name>), <unchanged|modified>` and `# <<< end <ID>`. The ID, original file and original line range in that marker are what section 5 refers to.
- Nothing in this experiment imports `dino-main` or `reference/`. `smoke_test.py` reads the original `DINOLoss`, `DINOHead` and `trunc_normal_` source text from `reference/` and runs it next to the adapted code on the same synthetic inputs.

## 2. The substantive adaptations, in plain English

**Backbone interface (model structure; necessary).** The original handles a TorchVision network by building it and replacing `backbone.fc` with an identity layer, which removes ResNet's classifier. MobileNetV2 has no `fc`: its classifier is `model.classifier` (Dropout 0.2, then Linear 1280 to 1000), so the original line would leave the classifier running and the "features" would be 1000 ImageNet class scores. `MobileNetV2Backbone` (new code in `dino.py`) keeps only `model.features`, then applies the same global average pooling as TorchVision's own forward pass, and flattens. Result: one 1280-value vector per image, for any input size. The ImageNet classifier and its dropout are not part of the SSL model at all.

**Feature dimension (model structure; necessary).** The original reads the embedding size from `fc.weight` (ResNet) or `embed_dim` (ViT). Here it is read from MobileNetV2's classifier input size (1280) and then checked by running the backbone on a 224 px and a 96 px input (in eval mode, so the check does not touch BatchNorm statistics). Three sizes must not be confused:

| Size | Value | What it is |
|---|---|---|
| MobileNetV2 feature dimension | 1280 | Output of the backbone after pooling. This is what fine-tuning uses. |
| DINO projection output dimension (`out_dim`) | 4096 | Size of the DINO head output that the softmax and the loss run over. A design choice of the SSL method; it is **not** the number of classes. |
| Downstream class count | 40 | Only used by the new linear classifier during supervised fine-tuning. |

**Classifier removal and export (model structure; necessary).** During SSL, the network is backbone, then DINO head (3-layer MLP, L2 normalization, weight-normalized last layer). After SSL, `export_backbone.py` keeps only the teacher's `features` (156 parameter tensors and 156 BatchNorm buffers: running means, variances and batch counters) and drops the DINO head. Fine-tuning then attaches a new Dropout 0.2 + Linear 1280 to 40 head, exactly as in A0.

**Initialization (learning method; a protocol decision, not a change to DINO).** The original DINO starts from random weights. Here the student backbone starts from the same `IMAGENET1K_V2` file that A0 uses (checked by hash and against TorchVision's cached file). The DINO head is new (random, initialized exactly as in the original). The teacher is then made an exact copy of the student: every parameter and every BatchNorm buffer, as the original's `load_state_dict` does.

**CSV loading (data handling; necessary for the protocol).** The original reads a whole image folder with `ImageFolder`, which would include the validation images and returns class labels. Here the SSL dataset reads only the `relative_path` column of `ssl_training_data.csv` and returns only the list of views, so no label can reach the loss, the sampling, or checkpoint selection. A separate split audit (`ssl_data.audit_splits`) reads the labels and metadata only to check for leaks before training.

**Single process and BatchNorm (execution; necessary to run on one GPU without a launcher).**
- The original always initializes a distributed process group (NCCL), even for one GPU. Here there is no process group: no `DistributedSampler`, no `DistributedDataParallel`, no `all_reduce`.
- Center update: the original sums the teacher outputs across GPUs with `all_reduce` and divides by the global batch. With one process the local batch is the global batch, so dividing by the local batch size gives the same center.
- BatchNorm: for networks with BatchNorm, the original converts both networks to `SyncBatchNorm` and wraps the teacher in DDP. `SyncBatchNorm` only shares batch statistics between processes; with one process it computes the same statistics as ordinary `BatchNorm2d`, which MobileNetV2 already uses. So ordinary BatchNorm is kept.
- What the teacher's BatchNorm does (same as the original). The teacher is never switched to eval mode, so in every forward pass:
  - it normalizes with the statistics of the current batch of global views (128 images at batch 64);
  - it updates its own running means and variances with momentum 0.1, and its batch counter goes up by 1 per step.
- The EMA update covers parameters only, as in the original; the running statistics are not averaged from the student.
- The student runs the global views (one forward) and the local views (another forward), so its BatchNorm updates twice per step and its batch counter goes up by 2.
- `smoke_test.py` checks all of this: the counters, the running-mean formula on the first BatchNorm layer, and that the teacher's and student's statistics diverge.
- Turning off gradients for the teacher (`requires_grad = False`) is different from calling `teacher.eval()`; the code does the first and not the second, as the original does.

**API replacements (compatibility; numerical meaning preserved).**
- `torch.nn.utils.weight_norm` is replaced by `torch.nn.utils.parametrizations.weight_norm`: same weight g * v / ||v||, new parameter names. Initialization and outputs are checked bit-identical to the original.
- The original's own `trunc_normal_` copy is replaced by `torch.nn.init.trunc_normal_`: same algorithm, checked bit-identical.
- `torch.cuda.amp.GradScaler` and `autocast` are replaced by `torch.amp.GradScaler("cuda")` and `torch.autocast("cuda", float16)`: the same mixed precision.

**Augmentation (data handling; SAR-specific choices, so this is an adaptation of the original recipe).** Details and numbers are in README.md, section "SSL recipe". Briefly:
- kept from the original: crop sizes (224 and 96), global crop scale, 8 local crops, flip, brightness and contrast jitter, and the blur schedule;
- removed because they do nothing on a gray image: saturation, hue and random grayscale;
- turned off for SAR: solarization (it would invert the bright scattering returns);
- narrowed: the local crop scale, 0.15 to 0.4 instead of 0.05 to 0.4;
- matched to the baseline: bilinear instead of bicubic resizing;
- unchanged and identical to A0: gray-to-3-channel copying and ImageNet normalization.

**Hyperparameters (learning method settings; experimental choices, not code changes).** The algorithm is unchanged, but several values differ from the original's ViT-oriented defaults; README.md explains each one:
- `out_dim` is 4096 instead of 65536;
- the teacher momentum is 0.9995 instead of 0.996, following the original's own advice for small batches;
- the weight decay is a constant 0.04 instead of a ramp from 0.04 to 0.4;
- the batch is 64 on one GPU.

**Run records, resume and checkpoint (execution).**
- The checkpoint additionally stores the global step, a fingerprint of all schedules, the random-number states and the scaler state.
- Resume restarts at an epoch boundary and refuses to continue if any setting, the image list, or the rebuilt schedules differ.
- Each epoch's image order is drawn from `seed + epoch`, as `DistributedSampler.set_epoch` does in the original, so it does not depend on whether the run was resumed.

## 3. Not copied (not used by this experiment)

| Original component | Why it is not here |
|---|---|
| `vision_transformer.py`: ViT, attention, patch embedding, `drop_path` | The backbone is MobileNetV2. |
| `main_dino.py`: argument parser, `torch.hub` XCiT list, `ImageFolder` data loading, `DistributedSampler`, `restart_from_checkpoint`, `MetricLogger` logging, `log.txt` | Replaced by the JSON config, the CSV-based SSL dataset, a per-epoch seeded shuffle, the explicit resume code in `train_ssl.py`, and `history.csv` / `run_info.json`. None of these change the learning algorithm. The Torch Hub architecture list is not used, so no network call is made. |
| `utils.py`: distributed helpers (`init_distributed_mode`, `get_world_size`, `save_on_master`, `setup_for_distributed`, `reduce_dict`), `MetricLogger`, `SmoothedValue`, `LARS`, `has_batchnorms`, `load_pretrained_weights` (downloads), evaluation helpers (k-NN, PCA, retrieval) | Single process, no downloads, and evaluation is the supervised fine-tuning protocol of A0. |
| `eval_*.py`, `hubconf.py`, `video_generation.py`, `visualize_attention.py`, `run_with_submitit.py` | Not needed for training; the downstream evaluation is the A0 fine-tuning pipeline. |

## 4. Downstream code versus the baseline

The supervised stage is a copy of `Experiments/Baseline` (originals kept in `reference/baseline_original/`). Training, preprocessing, the model head, the optimizer, the schedule, the metrics and the selection rule are unchanged. Every difference:

| File | Difference | Can it change results? |
|---|---|---|
| `train_supervised.py` (from `train.py`) | Imports this folder's `paths.py` instead of adding the repository root to `sys.path`; default output folder `outputs/finetune/`; `--backbone FILE` / `--init imagenet` instead of editing the config | No |
| `train_supervised.py` | `set_seed(seed)` is called again right before the model is built. In the baseline nothing between the first `set_seed` and `build_model` uses random numbers (reading CSVs and images does not), so the random state at that point is identical; the reset guarantees it stays so | No (same random state as A0) |
| `train_supervised.py` | Runs the split audit, saves copies of the labeled, validation and class-mapping CSVs in `manifests/`, and records the backbone file's SHA-256 and its full SSL record in `run_info.json` | No |
| `evaluate.py` | Reads the class mapping and validation list from the run's own `manifests/` copies and refuses to run if their hashes differ from those recorded at training time, or if the validation images differ; a smoke run's validation uses the same subset it was trained with | No (same images and preprocessing) |
| `data.py` | `load_class_names` and `read_manifest` accept a folder (default: the original location) | No |
| `plots.py` | New `plot_ssl_history` for the SSL charts; doc string paths | No |
| `config_finetune.json` (from `config.json`) | Run name `A1_dino_mobilenetv2`; `init` is set on the command line; every training value is identical | No |
| `model.py`, `engine.py`, `utils.py` | Identical copies | No |

The exact diffs are listed at the end of section 5 and stored as `diffs/baseline_*.diff`.

Parity check (2026-10-13): `train_supervised.py` with ImageNet initialization and the baseline's own `train.py` were run side by side on the CPU (smoke size: 8 images per class, 2 epochs, seed 0, separate processes, outputs outside the repository). The training histories were identical to every printed digit, and the saved `best.pt` weights were bit-identical (314 of 314 tensors). So the copied downstream code reproduces A0 exactly, and no correction was needed. If a correction ever changes downstream numbers, A0 must be rerun with the same correction before comparing.

## 5. Line-by-line record (generated)

<!-- BEGIN GENERATED by tools/make_change_record.py -->
### Copied DINO blocks

| ID | Original | Component | Destination | Status | Changed hunks | Exact diff |
|---|---|---|---|---|---|---|
| E1 | `utils.py` L36-L54 | GaussianBlur | `dino.py` L66-L84 | unchanged | 0 | none |
| E2 | `utils.py` L57-L68 | Solarization | `dino.py` L89-L100 | unchanged | 0 | none |
| E3 | `utils.py` L132-L141 | clip_gradients | `dino.py` L105-L114 | unchanged | 0 | none |
| E4 | `utils.py` L144-L149 | cancel_gradients_last_layer | `dino.py` L119-L124 | unchanged | 0 | none |
| E5 | `utils.py` L187-L198 | cosine_scheduler | `dino.py` L129-L140 | unchanged | 0 | none |
| E6 | `utils.py` L594-L629 | MultiCropWrapper | `dino.py` L145-L182 | modified | 3 | `diffs/dino_E6_MultiCropWrapper.diff` |
| E7 | `utils.py` L632-L643 | get_params_groups | `dino.py` L187-L198 | unchanged | 0 | none |
| E8 | `vision_transformer.py` L257-L291 | DINOHead | `dino.py` L204-L238 | modified | 3 | `diffs/dino_E8_DINOHead.diff` |
| E9 | `main_dino.py` L363-L416 | DINOLoss | `dino.py` L244-L297 | modified | 1 | `diffs/dino_E9_DINOLoss.diff` |
| E10 | `main_dino.py` L419-L464 | DataAugmentationDINO | `dino.py` L302-L348 | modified | 14 | `diffs/dino_E10_DataAugmentationDINO.diff` |
| E11 | `main_dino.py` L182-L212 | build student and teacher | `train_ssl.py` L126-L145 | modified | 5 | `diffs/dino_E11_build_student_and_teacher.diff` |
| E12 | `main_dino.py` L214-L251 | loss, optimizer, schedules | `train_ssl.py` L152-L190 | modified | 9 | `diffs/dino_E12_loss_optimizer_schedules.diff` |
| E13 | `main_dino.py` L301-L360 | train_one_epoch | `train_ssl.py` L203-L267 | modified | 9 | `diffs/dino_E13_train_one_epoch.diff` |
| E14 | `main_dino.py` L278-L290 | checkpoint contents | `train_ssl.py` L464-L478 | modified | 3 | `diffs/dino_E14_checkpoint_contents.diff` |

### Every changed hunk

Line numbers: original = line in `reference/dino-main_original/<file>`; destination = line in this folder. Categories: C compatibility (library/API), D data handling, M model structure, X execution (single process, device, logging, records), L learning method. No hunk is in category L: the learning mechanics are unchanged.

| Hunk | Original | Destination | Change | Original text (deleted) | New text (added) | Category | Why |
|---|---|---|---|---|---|---|---|
| E6.1 | `utils.py` L605-L606 | `dino.py` L156 | replaced | `# disable layers dedicated to ImageNet labels classification`<br>`backbone.fc, backbone.head = nn.Identity(), nn.Identity()` | `# the backbone (MobileNetV2Backbone) already has no ImageNet classifier` | M | The original disabled the ImageNet classifier by setting backbone.fc and backbone.head to Identity. MobileNetV2 has no fc, so its classifier (Dropout + Linear 1280->1000) would still run and return 1000 ImageNet logits. The backbone passed in here (MobileNetV2Backbone) already stops at global average pooling, so there is nothing to disable. |
| E6.2 | `utils.py` after L608 | `dino.py` L159 | added | (nothing) | `self.last_features = None` | X | Adds an attribute that holds the last pooled backbone features, for the collapse diagnostics only. |
| E6.3 | `utils.py` after L627 | `dino.py` L179-L180 | added | (nothing) | `# keep the pooled backbone features (no gradient) for the collapse diagnostics`<br>`self.last_features = output.detach()` | X | Stores the pooled features (detached: no gradient flows through this copy) after the loop over resolutions. Outputs and gradients are unchanged. |
| E8.1 | `vision_transformer.py` L276-L277 | `dino.py` L223-L224 | replaced | `self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))`<br>`self.last_layer.weight_g.data.fill_(1)` | `self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))`<br>`self.last_layer.parametrizations.weight.original0.data.fill_(1)` | C | torch.nn.utils.weight_norm is deprecated in torch 2.10 (FutureWarning). torch.nn.utils.parametrizations.weight_norm computes the same weight = g * v / ||v|| along dim 0. Only the parameter names change (weight_g -> parametrizations.weight.original0, weight_v -> original1). They still contain 'last_layer', so freeze_last_layer works as before. smoke_test.py checks that the initialization and outputs are identical to the original for the same seed. |
| E8.2 | `vision_transformer.py` L279 | `dino.py` L226 | replaced | `self.last_layer.weight_g.requires_grad = False` | `self.last_layer.parametrizations.weight.original0.requires_grad = False` | C | Same API change: the norm scale g is now named original0. As in the original, it is fixed at 1 and not trained when norm_last_layer is true. |
| E8.3 | `vision_transformer.py` L283 | `dino.py` L230 | replaced | `trunc_normal_(m.weight, std=.02)` | `nn.init.trunc_normal_(m.weight, std=.02)` | C | The original used its own copy of trunc_normal_ (in its utils.py, itself copied from PyTorch). torch.nn.init.trunc_normal_ is the same algorithm with the same defaults (a=-2, b=2); smoke_test.py checks that the initialization is bit-identical. |
| E9.1 | `main_dino.py` L412-L413 | `dino.py` L293-L294 | replaced | `dist.all_reduce(batch_center)`<br>`batch_center = batch_center / (len(teacher_output) * dist.get_world_size())` | `# single process: the whole batch is on this GPU, so no all_reduce and world size 1`<br>`batch_center = batch_center / len(teacher_output)` | X | Single process. The original summed the teacher outputs over all GPUs with dist.all_reduce and divided by (batch per GPU x world size). With one process the local sum is the whole sum and the world size is 1, so the center is numerically the same, without needing a process group. smoke_test.py checks the loss and center against the original code run with world size 1. |
| E10.1 | `main_dino.py` L420 | `dino.py` L303 | replaced | `def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):` | `def __init__(self, args):` | D | All settings come from the config (crop sizes, crop scales, flip, jitter, blur, solarization, normalization) instead of being hard-coded, so every augmentation choice is visible and recorded in each run's config.json. |
| E10.2 | `main_dino.py` L422 | `dino.py` L305 | replaced | `transforms.RandomHorizontalFlip(p=0.5),` | `transforms.RandomHorizontalFlip(p=args.hflip_p),` | D | Flip probability from the config: 0.5, the same as the original and the A0 baseline. |
| E10.3 | `main_dino.py` L424-L425 | `dino.py` L307-L308 | replaced | `[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],`<br>`p=0.8` | `[transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast)],`<br>`p=args.color_jitter_p` | D | SAR chips have one channel. Brightness and contrast jitter keep the original strengths (0.4, 0.4) and probability (0.8), now from the config; they act as radar gain changes. Saturation (0.2) and hue (0.1) are removed: on a gray image copied into 3 identical channels they do nothing. |
| E10.4 | `main_dino.py` L427 | `dino.py` (deleted) | deleted | `transforms.RandomGrayscale(p=0.2),` | (nothing) | D | RandomGrayscale (p 0.2) removed: the image is already gray, so it does nothing. |
| E10.5 | `main_dino.py` after L429 | `dino.py` L312 | added | (nothing) | `transforms.Grayscale(num_output_channels=3),` | D | Grayscale(3) copies the one SAR channel into R, G and B, exactly as the baseline does, so the ImageNet-pretrained first layer receives the same kind of input in SSL and in fine-tuning. |
| E10.6 | `main_dino.py` L431 | `dino.py` L314 | replaced | `transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),` | `transforms.Normalize(args.normalize_mean, args.normalize_std),` | D | Normalization values from the config: the same ImageNet mean/std as the original and the baseline. |
| E10.7 | `main_dino.py` after L432 | `dino.py` L316 | added | (nothing) | `interpolation = transforms.InterpolationMode.BILINEAR` | D | Bilinear interpolation (with antialias) instead of bicubic, to match the baseline's resize. A SAR-specific choice for consistency with fine-tuning, not a compatibility requirement. |
| E10.8 | `main_dino.py` L436 | `dino.py` L320 | replaced | `transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),` | `transforms.RandomResizedCrop(args.global_crop_size, scale=args.global_crops_scale, interpolation=interpolation, antialias=True),` | D | First global crop: size and scale from the config (224 px and 0.4-1.0: the original values), bilinear interpolation (see above). |
| E10.9 | `main_dino.py` L438 | `dino.py` L322 | replaced | `utils.GaussianBlur(1.0),` | `GaussianBlur(args.blur_p[0]),` | D | Blur probability from the config (1.0 for the first global view, as in the original). GaussianBlur now lives in dino.py. |
| E10.10 | `main_dino.py` L443 | `dino.py` L327 | replaced | `transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),` | `transforms.RandomResizedCrop(args.global_crop_size, scale=args.global_crops_scale, interpolation=interpolation, antialias=True),` | D | Second global crop: same as the first. |
| E10.11 | `main_dino.py` L445-L446 | `dino.py` L329-L330 | replaced | `utils.GaussianBlur(0.1),`<br>`utils.Solarization(0.2),` | `GaussianBlur(args.blur_p[1]),`<br>`Solarization(args.solarization_p),` | D | Blur probability from the config (0.1, as in the original). Solarization probability from the config, set to 0 for SAR: it inverts every pixel above 128, i.e. exactly the bright scattering returns that identify a vehicle in these mostly dark chips. The transform itself is unchanged. |
| E10.12 | `main_dino.py` L450 | `dino.py` L334 | replaced | `self.local_crops_number = local_crops_number` | `self.local_crops_number = args.local_crops_number` | D | Number of local crops from the config (8, the original default). |
| E10.13 | `main_dino.py` L452 | `dino.py` L336 | replaced | `transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),` | `transforms.RandomResizedCrop(args.local_crop_size, scale=args.local_crops_scale, interpolation=interpolation, antialias=True),` | D | Local crops: size from the config (96 px, as in the original); scale 0.15-0.4 instead of 0.05-0.4 (SAR-specific: with 0.05 about 14% of local crops contained no part of the vehicle; with 0.15 about 7%; see README); bilinear interpolation. |
| E10.14 | `main_dino.py` L454 | `dino.py` L338 | replaced | `utils.GaussianBlur(p=0.5),` | `GaussianBlur(p=args.blur_p[2]),` | D | Blur probability from the config (0.5, as in the original). |
| E11.1 | `main_dino.py` L183 | `train_ssl.py` L127 | replaced | `student = utils.MultiCropWrapper(student, DINOHead(` | `student = MultiCropWrapper(student, DINOHead(` | X | utils.MultiCropWrapper -> MultiCropWrapper (the class now lives in dino.py). The DINOHead arguments are unchanged. |
| E11.2 | `main_dino.py` L189 | `train_ssl.py` L133 | replaced | `teacher = utils.MultiCropWrapper(` | `teacher = MultiCropWrapper(` | X | Same for the teacher. |
| E11.3 | `main_dino.py` L194-L206 | `train_ssl.py` L138-L139 | replaced | `student, teacher = student.cuda(), teacher.cuda()`<br>`# synchronize batch norms (if any)`<br>`if utils.has_batchnorms(student):`<br>`student = nn.SyncBatchNorm.convert_sync_batchnorm(student)`<br>`teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)`<br>(blank line)<br>`# we need DDP wrapper to have synchro batch norms working...`<br>`teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])`<br>`teacher_without_ddp = teacher.module`<br>`else:`<br>`# teacher_without_ddp and teacher are the same thing`<br>`teacher_without_ddp = teacher`<br>`student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])` | `student, teacher = student.to(device), teacher.to(device)`<br>`# one process, one GPU: ordinary BatchNorm, no SyncBatchNorm and no DistributedDataParallel` | X | One process, one GPU. .to(device) instead of .cuda(). No SyncBatchNorm conversion: SyncBatchNorm only shares batch statistics between processes, so with one GPU it computes the same statistics as the ordinary BatchNorm that MobileNetV2 already has. No DistributedDataParallel wrappers (they only average gradients across processes and, for the teacher, broadcast buffers). teacher_without_ddp is therefore just the teacher. As in the original, the teacher is never put in eval mode. |
| E11.4 | `main_dino.py` L208 | `train_ssl.py` L141 | replaced | `teacher_without_ddp.load_state_dict(student.module.state_dict())` | `teacher.load_state_dict(student.state_dict())` | X | student.module -> student, teacher_without_ddp -> teacher (no DDP wrapper). Still a full state_dict copy: parameters and BatchNorm buffers. |
| E11.5 | `main_dino.py` L212 | `train_ssl.py` L145 | replaced | `print(f"Student and Teacher are built: they are both {args.arch} network.")` | `print(f"Student and Teacher are built: they are both MobileNetV2 networks.")` | X | The message names MobileNetV2 (there is no args.arch here). |
| E12.1 | `main_dino.py` L222 | `train_ssl.py` L160-L162 | replaced | `).cuda()` | `student_temp=args.student_temp,`<br>`center_momentum=args.center_momentum,`<br>`).to(device)` | X | DINOLoss receives student_temp and center_momentum from the config (the original used the class defaults; the config holds the same values, 0.1 and 0.9) so they are recorded. .to(device) instead of .cuda(). |
| E12.2 | `main_dino.py` L225 | `train_ssl.py` L165 | replaced | `params_groups = utils.get_params_groups(student)` | `params_groups = get_params_groups(student)` | X | get_params_groups now lives in dino.py (no utils. prefix). Same groups: biases and other 1-D parameters (BatchNorm weights) get no weight decay. |
| E12.3 | `main_dino.py` L230-L231 | `train_ssl.py` (deleted) | deleted | `elif args.optimizer == "lars":`<br>`optimizer = utils.LARS(params_groups)  # to use with convnet and large batches` | (nothing) | X | LARS option removed: the original suggests it for large multi-GPU convnet batches; it is not used here (the config accepts adamw or sgd). |
| E12.4 | `main_dino.py` L234-L235 | `train_ssl.py` L172-L173 | replaced | `if args.use_fp16:`<br>`fp16_scaler = torch.cuda.amp.GradScaler()` | `if args.use_fp16 and device.type == "cuda":`<br>`fp16_scaler = torch.amp.GradScaler("cuda")` | C | torch.cuda.amp.GradScaler is deprecated; torch.amp.GradScaler('cuda') is the same scaler. Mixed precision is only used on the GPU. |
| E12.5 | `main_dino.py` L238-L239 | `train_ssl.py` L176-L177 | replaced | `lr_schedule = utils.cosine_scheduler(`<br>`args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule` | `lr_schedule = cosine_scheduler(`<br>`args.lr * args.batch_size_per_gpu / 256.,  # linear scaling rule` | X | Linear scaling rule kept: effective lr = lr x total batch / 256. The original's total batch was batch_size_per_gpu x number of GPUs; with one GPU it is batch_size_per_gpu. cosine_scheduler now lives in dino.py. |
| E12.6 | `main_dino.py` L241 | `train_ssl.py` L179 | replaced | `args.epochs, len(data_loader),` | `args.epochs, niter_per_ep,` | X | len(data_loader) -> niter_per_ep, the same number (images // batch, because drop_last) computed once, so it can be recorded and compared on resume. |
| E12.7 | `main_dino.py` L244 | `train_ssl.py` L182 | replaced | `wd_schedule = utils.cosine_scheduler(` | `wd_schedule = cosine_scheduler(` | X | cosine_scheduler now lives in dino.py. |
| E12.8 | `main_dino.py` L247 | `train_ssl.py` L185 | replaced | `args.epochs, len(data_loader),` | `args.epochs, niter_per_ep,` | X | len(data_loader) -> niter_per_ep (same number). |
| E12.9 | `main_dino.py` L250-L251 | `train_ssl.py` L188-L190 | replaced | `momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,`<br>`args.epochs, len(data_loader))` | `momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,`<br>`args.epochs, niter_per_ep)`<br>`print(f"Loss, optimizer and schedulers ready.")` | X | cosine_scheduler from dino.py; niter_per_ep (same number). |
| E13.1 | `main_dino.py` L301 | `train_ssl.py` L203 | replaced | `def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,` | `def train_one_epoch(student, teacher, dino_loss, data_loader,` | X | teacher_without_ddp removed from the arguments (no DDP). |
| E13.2 | `main_dino.py` L303-L306 | `train_ssl.py` L205-L208 | replaced | `fp16_scaler, args):`<br>`metric_logger = utils.MetricLogger(delimiter="  ")`<br>`header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)`<br>`for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):` | `fp16_scaler, args, device, timing=None):`<br>`sums, t0 = {}, time.time()`<br>`for it, images in enumerate(data_loader):`<br>`step = it + 1` | X | device and an optional timing callback are passed in (the callback prints the measured speed and peak GPU memory early in the run). MetricLogger replaced by plain running sums. The loader yields only the list of views, because the SSL dataset returns no label; `step` counts batches for printing. The global iteration index `it` is computed exactly as before. |
| E13.3 | `main_dino.py` L315 | `train_ssl.py` L217-L218 | replaced | `images = [im.cuda(non_blocking=True) for im in images]` | `images = [im.to(device, non_blocking=True) for im in images]`<br>`center = dino_loss.center.clone()  # the center this step uses, for the diagnostics` | X | .to(device) instead of .cuda(). The center is copied before the loss updates it, so the diagnostics use the center that this step actually used. |
| E13.4 | `main_dino.py` L317 | `train_ssl.py` L220 | replaced | `with torch.cuda.amp.autocast(fp16_scaler is not None):` | `with torch.autocast(device.type, dtype=torch.float16, enabled=fp16_scaler is not None):` | C | torch.cuda.amp.autocast is deprecated; torch.autocast(device, dtype=float16) is the same. |
| E13.5 | `main_dino.py` L323-L324 | `train_ssl.py` L226 | replaced | `print("Loss is {}, stopping training".format(loss.item()), force=True)`<br>`sys.exit(1)` | `raise RuntimeError("Loss is {}, stopping training".format(loss.item()))` | X | A non-finite loss raises an error (the run is then recorded as failed in run_info.json) instead of print(force=True) + sys.exit; force= only exists in the original's distributed print. |
| E13.6 | `main_dino.py` L332-L334 | `train_ssl.py` L234-L236 | replaced | `param_norms = utils.clip_gradients(student, args.clip_grad)`<br>`utils.cancel_gradients_last_layer(epoch, student,`<br>`args.freeze_last_layer)` | `param_norms = clip_gradients(student, args.clip_grad)`<br>`cancel_gradients_last_layer(epoch, student,`<br>`args.freeze_last_layer)` | X | clip_gradients and cancel_gradients_last_layer now live in dino.py (no utils. prefix). Same order: backward, clip, cancel last-layer gradients during the first epoch(s), step. |
| E13.7 | `main_dino.py` L340-L342 | `train_ssl.py` L242-L244 | replaced | `param_norms = utils.clip_gradients(student, args.clip_grad)`<br>`utils.cancel_gradients_last_layer(epoch, student,`<br>`args.freeze_last_layer)` | `param_norms = clip_gradients(student, args.clip_grad)`<br>`cancel_gradients_last_layer(epoch, student,`<br>`args.freeze_last_layer)` | X | Same, in the mixed-precision branch (unscale, clip, cancel, step, update: same order). |
| E13.8 | `main_dino.py` L349 | `train_ssl.py` L251 | replaced | `for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):` | `for param_q, param_k in zip(student.parameters(), teacher.parameters()):` | X | student.module -> student, teacher_without_ddp -> teacher (no DDP). The EMA still runs over parameters only, exactly as in the original: BatchNorm running statistics are not averaged; the teacher updates them itself in its forward passes. |
| E13.9 | `main_dino.py` L353-L360 | `train_ssl.py` L255-L267 | replaced | `torch.cuda.synchronize()`<br>`metric_logger.update(loss=loss.item())`<br>`metric_logger.update(lr=optimizer.param_groups[0]["lr"])`<br>`metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])`<br>`# gather the stats from all processes`<br>`metric_logger.synchronize_between_processes()`<br>`print("Averaged stats:", metric_logger)`<br>`return {k: meter.global_avg for k, meter in metric_logger.meters.items()}` | `diag = collapse_diagnostics(teacher_output, center, dino_loss.teacher_temp_schedule[epoch],`<br>`teacher.last_features, len(images[0]))`<br>`for k, v in {"loss": loss.item(), **diag}.items():`<br>`sums[k] = sums.get(k, 0.0) + v`<br>`if timing is not None:`<br>`timing(step, len(data_loader))`<br>`if args.log_every_steps and (step % args.log_every_steps == 0 or step == len(data_loader)):`<br>`print(f"    step {step:5d}/{len(data_loader)}  loss {sums['loss'] / step:.4f}  "`<br>`f"lr {optimizer.param_groups[0]['lr']:.2e}  teacher entropy {sums['teacher_entropy'] / step:.3f}  "`<br>`f"{time.time() - t0:.0f}s", flush=True)`<br>`return {k: v / len(data_loader) for k, v in sums.items()} \| {`<br>`"lr": optimizer.param_groups[0]["lr"], "wd": optimizer.param_groups[0]["weight_decay"],`<br>`"momentum": m, "teacher_temp": float(dino_loss.teacher_temp_schedule[epoch])}` | X | torch.cuda.synchronize and MetricLogger replaced by the collapse diagnostics, running sums and progress printing (loss.item() already waits for the GPU). The returned dictionary also reports the lr, weight decay, momentum and teacher temperature used. No effect on training. |
| E14.1 | `main_dino.py` L283 | `train_ssl.py` L469 | replaced | `'args': args,` | `'args': vars(args),` | X | args saved as a plain dict (the flat config) so it can be compared on resume. |
| E14.2 | `main_dino.py` after L284 | `train_ssl.py` L471-L474 | added | (nothing) | `'fp16_scaler': fp16_scaler.state_dict() if fp16_scaler is not None else None,`<br>`'global_step': (epoch + 1) * niter_per_ep,`<br>`'schedule_info': schedule_info,`<br>`'rng_state': rng_state(),` | X | Added for exact epoch-boundary resume: the scaler state (None when not used), the global step, the schedule fingerprint and the random number generator states. |
| E14.3 | `main_dino.py` L286-L290 | `train_ssl.py` L476-L478 | replaced | `if fp16_scaler is not None:`<br>`save_dict['fp16_scaler'] = fp16_scaler.state_dict()`<br>`utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))`<br>`if args.saveckp_freq and epoch % args.saveckp_freq == 0:`<br>`utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))` | `tmp = run_dir / "checkpoint.pth.tmp"`<br>`torch.save(save_dict, tmp)`<br>`os.replace(tmp, run_dir / "checkpoint.pth")  # only replaced once the new file is complete` | X | Written to a temporary file and then renamed, so an interruption never leaves a half-written checkpoint. No periodic copies (saveckp_freq): the final epoch is the checkpoint chosen in advance, and DINO loss is not used to pick another one. |

### Downstream files copied from the baseline

| Baseline file | Copy | Lines added / removed | Exact diff |
|---|---|---|---|
| `Baseline/data.py` | `data.py` | +6 / -6 | `diffs/baseline_data.py.diff` |
| `Baseline/model.py` | `model.py` | identical | none |
| `Baseline/engine.py` | `engine.py` | identical | none |
| `Baseline/utils.py` | `utils.py` | identical | none |
| `Baseline/plots.py` | `plots.py` | +55 / -1 | `diffs/baseline_plots.py.diff` |
| `Baseline/evaluate.py` | `evaluate.py` | +34 / -12 | `diffs/baseline_evaluate.py.diff` |
| `Baseline/train.py` | `train_supervised.py` | +53 / -19 | `diffs/baseline_train_supervised.py.diff` |
| `Baseline/config.json` | `config_finetune.json` | +5 / -4 | `diffs/baseline_config_finetune.json.diff` |
<!-- END GENERATED -->
