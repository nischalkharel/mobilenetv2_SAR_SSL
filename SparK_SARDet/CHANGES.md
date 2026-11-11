# SparK_SARDet: what was brought in, what changed, and why

Port of the ATRNet-STAR A3 experiment (`mobilenetv2_SAR_SSL/Experiments/mobileNetv2_SparK/`, smoke-tested
2026-10-27, reviewed line by line) onto the SARDet-100K suite's shared engine (`shared/ssl_engine.py`). The method
(masked image modelling with a sparse MobileNetV2 encoder and the authors' LightDecoder) is unchanged; the data
pipeline, run records, checkpoints, resume and export are now the suite's shared code. Paper: Tian et al., "Designing
BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling", ICLR 2023 (arXiv 2301.03580 v2). Code:
keyu-tian/SparK, commit a63e386 (MIT; `spark_lib/LICENSE`).

## 1. Files

| File | Origin | What differs |
|---|---|---|
| `spark_lib/LICENSE`, `lamb.py`, `lr_control.py` | the authors' `LICENSE`, `pretrain/utils/lamb.py`, `pretrain/utils/lr_control.py` | nothing (byte-identical to the authors' files and to the ATRNet-STAR copies; sha256 in section 5) |
| `spark_lib/encoder.py`, `spark.py`, `decoder.py` | the authors' `pretrain/encoder.py`, `spark.py`, `decoder.py` | ONLY the import lines marked `# CHANGED (A3)` (timm is not installed; package-relative imports; `is_pow2n` inlined). Byte-identical to the ATRNet-STAR copies, which the A3 smoke test compared line by line with the originals |
| `spark_lib/__init__.py` | new | docstring only |
| `spark_model.py` | ATRNet-STAR `spark_model.py` | `MobileNetV2Hierarchical` loads the ImageNet file through `shared.backbone.load_imagenet_mobilenet_v2()`; `build_spark` reads the view size from the shared key `global_size`; the view class `DataAugmentationSparK` and `to_network_input` are REMOVED (views come from `shared/ssl_data.py`); new `encoder_features_state()` (the export, prefix `sparse_encoder.sp_cnn.features.` stripped) and `feature_shapes()`; the reconstruction preview moved here from the old `preview_views.py` and now takes rows of the shared pool (centre 224 crop of the detector-resized image). `SparKMobileNetV2.forward`, `feature_diagnostics`, `count_parameters`: unchanged |
| `train_ssl.py` | ATRNet-STAR `train_ssl.py` (`train_one_epoch`, optimizer/schedule setup, `use_true_fp32`, `validate`) | wrapped into `class SparKMethod` for the engine; everything the engine does (config merge, pool, loader, run folder, checkpoint, resume, history, charts, cost, export) is deleted here. The per-step computation is the A3 loop verbatim: `lr_wd_annealing` -> forward -> `backward` -> LAMB `step()` -> diagnostics |
| `config_ssl.json` | ATRNet-STAR `config_ssl.json` | only method-specific keys remain (`spark_views`, `spark`, `optim`); `setup`, `views`, `batch_size`, `epochs` are the shared block now |
| `smoke_test.py` | mirrors `DINO_SARDet/smoke_test.py` | plus `check_spark_sanity()` (section 4) |
| `run_finetune.py`, `config_finetune.json` | copies of DINO_SARDet's | run name `spark_mobilenetv2` |

## 2. What changed in the recipe and why (one table)

| Item | ATRNet-STAR A3 | SparK_SARDet | Why |
|---|---|---|---|
| View | 1 per image, `RandomResizedCrop(224, scale 0.67-1.0)` + flip 0.5 on the 128 px chip, bilinear | 1 per image: the SHARED view recipe (`shared/ssl_common.json`): image resized by the detector rule (512), square crop of 224 x zoom in [0.75, 1.33] at a random position, resized to 224, flip 0.5, brightness/contrast 0.4 (p 0.8), **blur probability 0** (`spark_views.blur_p`) | DESIGN.md section 4: one view recipe for every method, so that the object scale matches detection. The crop is no longer the authors' RandomResizedCrop (the shared zoom-jitter crop replaces it). The authors use "only the minimal augmentation" (crop + flip), so the method-specific blur probability is 0; the shared brightness/contrast jitter applies to every method and is kept (per-patch normalisation of the target removes a global brightness/contrast change anyway) |
| Input normalisation | gray -> 3 channels -> /255 -> ImageNet mean/std on the GPU (`to_network_input` of the A3 folder) | `shared.ssl_data.to_network_input`, same arithmetic | shared code |
| Pool, loader, seeds | A3's `SSLDataset` + pixel cache, loader seeded `seed + epoch` | `shared.ssl_data.SSLPoolDataset` on the audited SSL pool, loader seeded by the engine (`seed * 100003 + epoch`), seed = split - 1 | shared code; no pixel cache |
| Batch, epochs | 64, 100 | shared block: 64, 100 (102,100 steps on ATRNet; here `len(pool) // 64` steps per epoch) | DESIGN.md "equal epochs" |
| Export | every 5 epochs + final, own `export_backbone.py` | FINAL epoch only, `shared.backbone.export_backbone` (strict load into torchvision `features`, 312 tensors, SHA-256 recorded) | the suite exports the final epoch, never a selected one; intermediate exports were for the accuracy-vs-epoch analysis only |
| Resume | own checkpoint + `repair_after_stop` (history row / export rebuilt from the checkpoint) | the engine's per-epoch checkpoint and exact resume (settings, manifest and schedule fingerprint checked; RNG restored) | shared code. The engine writes `checkpoint.pth` before `history.csv`; a kill in that window would leave the row missing (noted in `NOTES_FOR_MAINTAINER.md`) |
| Preserved RNG around the preview | `preserved_rng()` in A3 | `ssl_engine.preserved_rng()` around `save_reconstruction_preview` (every epoch) | the masks come from the global torch generator; an eval forward or a fresh `mobilenet_v2()` would shift it and a resumed run would draw different masks than an uninterrupted one (A3 pitfall, kept). The engine's own `fresh_features()` runs only after the last epoch |
| `densify_norm` | `"bn"` explicit | `"bn"` explicit, validated | the authors' `arg_util.py` leaves it EMPTY (no normalisation) and `beta2 = 0.0` for any model name without "resnet"/"convnext" (A3 pitfall, kept): both are set explicitly in `config_ssl.json` and checked in `SparKMethod.__init__` |
| Unused buffers | `imn_m`, `imn_s` zeroed | same | `torch.empty` buffers would make checkpoints differ for no reason |
| Windows GPU memory cap | 11 GB cap in A3 | not ported | engine concern; the PC's 12 GB card is for smoke tests, the full run goes to CRC |
| Reconstruction preview | at every export (every 5 epochs), `charts/reconstruction_epochNNN.png` | every epoch, `<run>/reconstruction_epochNNN.png`, 6 fixed pool images (seed 0, centre 224 crop) with fixed masks (seed 0), model in eval mode | the task asked for one per epoch; same drawing |
| Param-group listing | `param_groups.txt` | same (also holds the SparK densify and LAMB start-up lines) | the authors' code prints every parameter name |

Everything else (encoder adapter and feature maps, sparse conversion, mask, densify, decoder, loss, SAR target, LAMB,
warm-up/cosine schedule, weight-decay schedule, gradient clipping, fp32 with TF32 off, diagnostics) is the ATRNet-STAR
A3 recipe, itself the authors' ResNet recipe with the choices listed in the old `SPARK_CHANGES.md` section 3.

## 3. Exact hyperparameters (`config_ssl.json` + `shared/ssl_common.json`)

| Setting | Value | Origin |
|---|---|---|
| Initialisation | encoder = torchvision MobileNetV2 IMAGENET1K_V2 file (`shared/paths.py`, same file as every method); decoder side (densify BN + projections, 4 mask tokens, LightDecoder) random | protocol |
| Encoder maps | `features[3]` 24@56, `[6]` 32@28, `[13]` 96@14, `[18]` 1280@7 (strides 4-32) | A3 choice: last layer of each resolution; `[18]` so the 320 -> 1280 layer is trained |
| Sparse conversion | every Conv2d -> `SparseConv2d` (52), every BatchNorm2d -> `SparseBatchNorm2d` (52, BN1d over visible positions); depthwise convs keep `groups` | authors' `dense_model_to_sparse` |
| View | 1 global view, 224 px, blur_p 0 (shared: zoom 0.75-1.33, flip 0.5, brightness/contrast 0.4 p 0.8, no solarisation) | DESIGN.md section 4 |
| Mask | 7 x 7 patches of 32 px; `mask_ratio` 0.6 -> `len_keep = round(49 x 0.4) = 20` visible, 29 hidden (59.2 %) per image | authors (SimMIM values) |
| Densify | `densify_norm` "bn" (sparse BN), 4 mask tokens (1280, 96, 32, 24), projections 1x1 1280->768, 3x3 96->384, 32->192, 24->96 | authors; `bn` set explicitly |
| Decoder | `LightDecoder(32, width=768)`: 5 UNet blocks 768 -> 24, last 1x1 conv -> `recon_channels` = **1** output channel (trunc-normal 0.02 init) | authors' width; 1 channel = A3 change |
| Target and loss | channel 0 of the normalised input (= the SAR channel), per-patch normalised (mean 0, std 1, eps 1e-6), MSE averaged over the 29 hidden patches only | authors' loss on the SAR channel |
| Optimizer | `TheSameAsTimmLAMB`, betas (0.9, 0.95), eps 1e-6, bias correction, global gradient norm clipped at 5.0 inside `step()` | authors (`main.py` L79, `beta2` set explicitly) |
| Parameter groups | no weight decay on biases, 1-D tensors and `mask_token`s (`get_param_groups(nowd_keys={'cls_token','pos_embed','mask_token','gamma'})`) | authors (`main.py` L75) |
| Learning rate | peak = 2e-4 x 64 / 256 = **5e-5**; warm-up **2.5 epochs** linear from 0.005 x peak; then cosine to 0.001 x peak (per iteration, `lr_wd_annealing`) | authors' rule; 40/1600 = 2.5 % of training |
| Weight decay | cosine 0.04 -> 0.2 over the whole run | authors |
| Precision | **true float32**: no autocast; `torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 = False` (set in `SparKMethod.build` on CUDA; PyTorch 2.14 defaults are cudnn True, matmul False) | A3 measurement (section 6) |
| Batch, epochs, seed | 64, 100, seed = split - 1 | shared block |
| Export | encoder `features` of the final epoch (312 tensors incl. BN buffers), via `shared.backbone.export_backbone` | DESIGN.md |
| Smoke overrides | `warmup_epochs` 0.5 (engine: 48 images, batch 8, 2 epochs -> 12 steps, 3 of them warm-up) | the full 2.5-epoch warm-up would exceed the smoke run |
| Logged per epoch | loss, feature_cosine, feature_std (over visible positions of the 1280@7 map), grad_norm (global, before clipping), lr, wd, seconds | diagnostics only |

Parameters (counted at build, printed by `describe()`): encoder 2,223,872 (exported), decoder 23,183,065 (width
768, 1 output channel), densify norms + projections 1,395,152, mask tokens 1,432; total 26,803,521 trainable (the
engine records the total as `num_trainable_parameters`).

## 4. Smoke-test checks specific to SparK (`check_spark_sanity`, CPU)

- the six `spark_lib` files are byte-identical to the ATRNet-STAR copies (skipped if that folder is absent);
- densify norms are sparse BN; every encoder conv / BN is the authors' sparse class; 17 depthwise convs keep `groups`;
- the mask hides exactly `round(49 x 0.6) = 29` patches per image, different per image; `len_keep` = 20;
- dense `MobileNetV2Hierarchical` and the sparse encoder agree on a fully visible input (eval, max diff < 1e-4);
  the four maps are 24@56, 32@28, 96@14, 1280@7;
- hidden positions are exactly 0 after every encoder layer; the loss on hidden patches is finite and positive;
- with `recon_channels = 3` the loss equals the authors' `SparK.forward` on the same input and mask;
  with `recon_channels = 1` the target is channel 0 of the input;
- the exported state has exactly the keys of `torchvision.mobilenet_v2().features.state_dict()` (parameters and
  BN buffers, 312 tensors), loads strictly, and equals the ImageNet file before training.
Then (as DINO): a 2-epoch smoke SSL run (asserts `reconstruction_epoch001/002.png`), stop-after-1 + resume, the
shared fine-tune smoke from the exported backbone, probes and visual panels.

## 5. Hashes of the authors' code as used (`spark_lib/`)

| File | sha256 |
|---|---|
| LICENSE | 7c30bb95eed6002f42ab1f68cb18a7c580e75a409bc3fc7e93fb1925b510abb7 |
| decoder.py | 93ebbeea55843208e7528391b41ca1fb4d349cb842e7d2609068bbb632ec2522 |
| encoder.py | 80c34ae1d1724eabf3800ad067e2a762c642586173f1d6c2ebc921ec61bc1ff3 |
| lamb.py | ea0d56d613b4b11fe937c82f279a62815fbd62b67fac160841c6d8c6de73d34c |
| lr_control.py | 336cf1d35d4bb5e4eac6e6df99f375a7c701cbd15643c204429d88c1ca4dd0be |
| spark.py | 75bc5b869178892a6471c1748cdb193b42981aca71ec387b3f360e064784fddb |

## 6. Precision flags (why TF32 is off)

Measured on ATRNet-STAR (8 SSL chips at initialisation, gradient error against a float64 reference): CPU float32
3 %; GPU float32 with TF32 off 11 %; GPU TF32 (PyTorch default for convolutions) 102 %; bf16 autocast 129 %. With
TF32 the encoder's gradient is mostly rounding noise (ImageNet normalisation maps the dark SAR background to about
-2.1 and BatchNorm follows every conv, so many gradients are small remainders of large cancelling sums). Not
re-measured on SARDet-100K. The flags in force are printed by `describe()` at the start of every run, stored in each
checkpoint (`precision`, checked on resume) and written into the exported backbone's metadata (`exported.precision`).
Cost: without TF32 an A100 is not faster than the PC's GPU for this model.

## 7. Smoke run on this PC (RTX 4070 SUPER 12 GB, torch 2.14.1+cu130, Python 3.13.4, 2026-11-11)

`python SparK_SARDet/smoke_test.py` passed end to end (shared checks, SparK sanity, 2-epoch SSL run with
`reconstruction_epoch001/002.png`, stop-after-1 + resume producing the same epoch-1 loss 0.8707 and epoch-2 loss
0.8589 as the uninterrupted run, export of 312 tensors, fine-tune smoke, probes and panels). Smoke loss 0.87 -> 0.86
(near 1.0 = predicting each patch's mean, as expected after 12 steps at lr <= 6e-6).

Measured training speed (synthetic uint8 input through the real method: schedule, forward, backward, LAMB step;
mean of steps 11-30, cudnn.benchmark on, fp32 with TF32 off, 11 GB allocator cap so Windows cannot spill into RAM):

| Batch | s/step | ms/image | peak GPU memory |
|---|---|---|---|
| 8 (smoke) | 0.130 | 16.2 | 3.11 GB |
| 16 | 0.168 | 10.5 | 3.84 GB |
| 32 | 0.270 | 8.4 | 6.32 GB |
| 64 (full run) | out of memory under the 11 GB cap on this card | | about 11-12 GB by extrapolation: fits a 40 GB A100 |

The full run (1,469 steps per epoch at batch 64 on the 94,034-image pool, 100 epochs = 146,900 steps) is for the
CRC A100; at roughly 8-9 ms/image without TF32 that is about 21-24 GPU-hours (not measured on an A100).
