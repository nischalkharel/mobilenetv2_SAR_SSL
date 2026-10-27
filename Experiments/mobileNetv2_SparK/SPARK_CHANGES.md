# What was taken from the SparK authors, and what changed

Reference files: `reference/SparK-main_original/` (untouched; identical to commit a63e386; hashes in `reference/SOURCE.md`). Paper: arXiv 2301.03580 v2. Line numbers refer to those reference files.

## 1. Reused without change, or with import lines only

| Our file | Original | What differs |
|---|---|---|
| `spark_lib/lamb.py` | `pretrain/utils/lamb.py` | nothing (byte-identical) |
| `spark_lib/lr_control.py` | `pretrain/utils/lr_control.py` | nothing (byte-identical) |
| `spark_lib/LICENSE` | `LICENSE` (MIT) | nothing (byte-identical) |
| `spark_lib/encoder.py` | `pretrain/encoder.py` | L9 only: `from timm.models.layers import DropPath` became `DropPath = None`. timm is not installed; only `SparseConvNeXtBlock` uses DropPath, never MobileNetV2 |
| `spark_lib/spark.py` | `pretrain/spark.py` | L13: `trunc_normal_` from `torch.nn.init` instead of timm (timm's is a copy of PyTorch's: same arguments, same sampling). L15-L16: package-relative imports |
| `spark_lib/decoder.py` | `pretrain/decoder.py` | L12: `trunc_normal_` from `torch.nn.init`. L14: `is_pow2n` defined in place, copied from `utils/misc.py` L30-L31 (that module needs pytz and tensorboard) |

New: `spark_lib/__init__.py` (a docstring only). Every changed line is marked `# CHANGED (A3)`. `smoke_test.py` (section 2) checks all of the above against the reference copies.

What these files do, unchanged:
- `encoder.py`: the sparse layers. A "sparse" conv is the ordinary dense conv followed by multiplying its output with the mask at that resolution, so hidden positions stay exactly 0 ("submanifold" behaviour simulated with dense ops; the README explains they avoid real sparse kernels for speed). Sparse BatchNorm normalizes only the visible positions (a BatchNorm1d over the gathered visible vectors). `dense_model_to_sparse()` replaces every Conv2d and BatchNorm2d of any CNN, copying weights, BatchNorm statistics and the conv's `groups`, so MobileNetV2's depthwise convs stay depthwise.
- `spark.py`: random patch mask, encoding, "densifying" (BatchNorm, then the empty positions filled with a learned mask token, then a projection to the decoder width) at 4 scales, decoding, and the loss (per-patch normalized pixels, mean squared error on hidden patches only).
- `decoder.py`: `LightDecoder`, the UNet-style decoder of the paper's Appendix A: 5 blocks, each a 2x up-sampling transposed conv followed by two 3x3 convs, each conv followed by BatchNorm, with ReLU6 between them; then a 1x1 conv to the output channels.
- `lamb.py` (LAMB optimizer, global gradient norm clipped at `max_grad_norm` inside `step()`), `lr_control.py` (warm-up + cosine learning rate, cosine weight decay, and the parameter grouping).

## 2. Written from the authors' code or text (new code, same computation)

| Our code | Source | Difference |
|---|---|---|
| `MobileNetV2Hierarchical` (`spark_model.py`) | `pretrain/models/custom.py` (the 3 functions a custom CNN needs), with `models/resnet.py` as the pattern | MobileNetV2 instead of ResNet: maps after `features[3]`, `[6]`, `[13]`, `[18]` (section 3, item 1) |
| `build_spark` | `main.py` L57-L62; `arg_util.py` L116-L117 sets `sbn = False` on one GPU | for `recon_channels = 1` the decoder's last 1x1 conv has 1 output channel instead of 3, initialized as `LightDecoder.initialize()` does |
| `SparKMobileNetV2.forward` | `spark.py` L87-L128, copied | target = the first `recon_channels` input channels (section 3, item 3); also returns the smallest encoder map and the mask, for logging. Smoke test: with `recon_channels = 3` the loss equals the authors' `SparK.forward` on the same input and mask |
| `DataAugmentationSparK` | `utils/imagenet.py` L66-L71 | bilinear with antialias instead of bicubic; gray uint8 out, normalization on the GPU (section 5) |
| optimizer and schedule (`train_ssl.py`) | `main.py` L75-L81 and L152 | the same calls: `get_param_groups(nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})`, `TheSameAsTimmLAMB(lr, weight_decay=0, betas=(0.9, beta2), max_grad_norm=5.0)`, `lr_wd_annealing()` before every step |
| export (`export_backbone.py`) | `main.py` L106 saves `sparse_encoder.sp_cnn.state_dict()` | only the keys of MobileNetV2's `features`, plus run metadata; checked by a strict load into torchvision's MobileNetV2 |

## 3. Choices we had to make (there is no official MobileNetV2 or SAR recipe)

1. **Which feature maps.** The last layer of each resolution, at strides 4, 8, 16, 32, as ResNet's 4 stages: `features[3]` (24 channels), `[6]` (32), `[13]` (96), `[18]` (1280). `[18]` is MobileNetV2's final 1x1 conv; stopping at `[17]` (320 channels) would leave the 320 -> 1280 layer out of SSL although it is exported and fine-tuned. The mask cell = one position of the 7 x 7 map = 32 px, as in the authors' code (the patch size equals the total downsampling).
2. **`densify_norm = "bn"` and LAMB `beta2 = 0.95`.** `arg_util.py` sets these only when the model name contains "resnet" or "convnext" (L123-L129). For any other name `densify_norm` stays empty (no normalization) and `beta2` stays 0.0 (LAMB with no averaging of squared gradients): a silent trap for custom CNNs. MobileNetV2 is a BatchNorm CNN like ResNet, so ResNet's values are set explicitly.
3. **Reconstruction target: the SAR channel (`recon_channels = 1`).** The authors reconstruct the 3 RGB channels. Our 3 input channels are one gray SAR channel copied 3 times and normalized with the per-channel ImageNet mean and std. Per-patch normalization then turns the constant channel offsets into part of the target: measured on 400 chips (19,600 patches), a median **41%** of each patch's target variance (90th percentile 85%; 72% on the flatter half of the patches, mostly background) is that fixed offset pattern, not SAR content (`preview_views.py`, `crop_and_mask_coverage.json`). Reconstructing channel 0 removes it: per-patch normalization ignores any shift and scale, so channel 0 normalized per patch is the SAR image normalized per patch (smoke test). Same loss formula; `recon_channels = 3` restores the authors' literal target.
4. **Precision: full float32, TF32 switched off.** The authors' code trains without mixed precision (`main.py` has no autocast), and the paper mentions training on 32 GB Tesla V100s (Section 4.2, "Efficiency"), which have no TF32. PyTorch uses TF32 by default for float32 convolutions on A100 and RTX 30/40 GPUs. Gradient error against a float64 reference, measured on 8 SSL images at initialization:

   | Precision | Whole gradient | Encoder conv weights (median) |
   |---|---|---|
   | CPU float32 | 3% | 3% |
   | GPU float32, TF32 off (used) | 11% | 9% |
   | GPU TF32 (PyTorch default) | 102% | 62% |
   | GPU bf16 autocast | 129% | 145% |

   With TF32 the encoder's gradient would be mostly rounding noise. Likely reason: ImageNet normalization maps the dark SAR background to about -2.1, and BatchNorm follows every conv, so many gradients are small remainders of large cancelling sums. The A1 (fp16) and A2 (bf16) runs may be affected too; not measured yet (`TODO.md`).
5. **Batch 64, peak learning rate 5e-5.** Batch as A1 and A2. The authors' rule `lr = base_lr x batch / 256` (arg_util.py L132; paper Section 4.1) with their `base_lr = 2e-4`.
6. **100 epochs, 2.5 warm-up epochs.** Epochs as A1 and A2 (Nischal's decision); the authors pre-train 1,600 (their ablation: 800 epochs cost 0.4 ImageNet points, Table 5). Warm-up: the authors' 40 of 1,600 epochs is the first 2.5% of training, kept as 2.5% (2,552 steps). Their warm-up starts at 0.005 x lr and the cosine ends at 0.001 x lr; weight decay rises 0.04 -> 0.2 (cosine): unchanged.
7. **Decoder width 768**, the authors' only setting ("used equally for all encoders", paper Appendix A). With MobileNetV2 the decoder (23.2 M parameters) is about 10 times the encoder (2.2 M): SSL compute is dominated by the decoder. It is discarded after SSL, so it does not change the exported backbone's size.
8. **Mask ratio 0.6, 32-px patches** (authors; SimMIM's values). 20 of 49 patches stay visible (round(49 x 0.4)), so 59.2% are hidden. On our chips a vehicle spans about 2.8 x 2.8 patches; with the crops and masks of training, a median 39% of the vehicle stays visible, at least half of it in 22% of views, and it is completely hidden in 0.5% (2,000 chips, 8,000 samples; XML boxes used only for this check).
9. **Crop scale 0.67 to 1.0 and horizontal flip** (authors: "only the minimal augmentation"). Crops keep at least 90% of the vehicle in 93% of views.
10. **ImageNet initialization of the encoder** (study protocol; A0, A1, A2 start from the same file). The decoder side starts random. The authors pre-train from scratch; their pretrain/README suggests starting from pre-trained weights for small datasets.
11. **No drop path.** The authors use stochastic depth 0.05 for ResNet-50 (`models/__init__.py` L36). MobileNetV2 has no such layers; adding them would change the architecture.
12. **Unused buffers zeroed.** `imn_m` and `imn_s` (`spark.py` L76-L77, "deprecated and would never be used") are `torch.empty`, i.e. uninitialized memory saved into every checkpoint; set to 0.
13. **Data order.** Our loader (a new shuffle per epoch from seed + epoch, `drop_last`: 1,021 steps per epoch, as A1 and A2) instead of the authors' infinite sampler that fills the last batch with repeats.
14. **Random state around checkpoints.** The masks come from PyTorch's global generator, as in the authors' code. Exports build a fresh MobileNetV2 (random initial weights), which used to shift the generator, so a resumed run drew different masks than an uninterrupted one (found by the smoke test). Everything after a checkpoint now runs with the generator state restored afterwards.
15. **Diagnostics** (logged only, never in the loss or the checkpoint choice): global gradient norm before clipping (as the authors log it), and the A1/A2 feature collapse checks computed over visible positions.

## 4. Downstream code

`data.py`, `model.py`, `engine.py`, `evaluate.py`, `train_supervised.py`, `utils.py`, `config_finetune.json`, `ssl_data.py` and `paths.py` are copies of the files in `Experiments/mobileNetv2_LeJEPA` (which copies the DINO experiment's, i.e. the A0 code plus the documented A1 changes, plus the `--train-csv` option for label budgets). Changes here: file paths, help texts and docstrings; `run_name` `A3_spark_mobilenetv2`; `ssl_data.flat_args` reads a `spark` config section. Fine-tuning settings are identical to A0, A1 and A2. `plots.py`: the SSL charts were replaced; the fine-tuning and evaluation charts are unchanged.

## 5. Engineering changes that do not change the recipe

- **Last input steps on the GPU** (as A2): the loader returns a uint8 gray view; `to_network_input` copies it into 3 channels, divides by 255 and normalizes on the GPU. Same operations as `Grayscale(3) -> ToTensor() -> Normalize()` in A0/A1/A2 (smoke test: largest difference 7e-7).
- **Parameter-group listing**: `get_param_groups` prints every parameter name; the text goes to `param_groups.txt` in the run folder instead of the log.
- **Reconstruction pictures**: at every export, 6 fixed chips with fixed masks go through the model in eval mode (`spark.py`'s own `vis=True` path; no statistics are updated) into `charts/reconstruction_epoch<NNN>.png`.
- **Safe against a stop at any moment** (from the independent code review): the checkpoint also stores its epoch's history row and cost, and resume rebuilds a missing history row, cost or due backbone export from it (the smoke test simulates such a stop). JSON records are written through a temporary file, so a crash cannot leave a half-written `run_info.json`. Each resume records its time, Git commit and machine.
