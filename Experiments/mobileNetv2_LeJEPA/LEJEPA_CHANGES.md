# What was taken from the LeJEPA authors, and what changed

Reference files: `reference/lejepa-main_original/` (untouched; hashes in `reference/SOURCE.md`). Paper: arXiv 2511.08544 v3.
Line numbers refer to those reference files.

## 1. Reused without any change

| Our file | Original | Check |
|---|---|---|
| `lejepa_lib/univariate/base.py` | `lejepa/univariate/base.py` | byte-identical (smoke test, section 2) |
| `lejepa_lib/univariate/epps_pulley.py` | `lejepa/univariate/epps_pulley.py` | byte-identical |
| `lejepa_lib/multivariate/slicing.py` | `lejepa/multivariate/slicing.py` | byte-identical |

New: the three `lejepa_lib/**/__init__.py` files. The originals import every statistical test in the package, and some need scipy (not installed, not needed). Ours import only `UnivariateTest`, `EppsPulley` and `SlicingUnivariateTest`.

SIGReg is built as in the README's quick start (`README.md` L105-L115): `SlicingUnivariateTest(univariate_test=EppsPulley(...), num_slices=1024)`. The README writes `EppsPulley(num_points=17)`, which raises a `TypeError` in the shipped code (the argument is `n_points`); we pass `n_points=17` and the library default `t_max=3`.

Smoke-test section 4 shows that the library's SIGReg gives the same value as the `SIGReg` class printed in `MINIMAL.md` (L57-L75) when both use the same random directions (3.157266 both).

## 2. Written from the authors' code or text (new code, same computation)

| Our code (`lejepa_model.py`) | Source | Difference |
|---|---|---|
| `lejepa_loss`: `agreement = (centers - proj).square().mean()` with `centers = proj[:n_global].mean(0)` | paper Section 5.1 listing; with all views global it is `MINIMAL.md` L172 `inv_loss = (proj.mean(0) - proj).square().mean()` | none in the value (smoke test, section 4). Computed in float32 outside autocast; MINIMAL computes it inside bf16 autocast |
| `sig = sigreg(proj)` on the (V, N, K) tensor | `MINIMAL.md` L173 | float32 instead of bf16 (same reason) |
| `total = sig * lamb + agreement * (1 - lamb)` | `MINIMAL.md` L174 | none |
| `LeJEPANet.proj = MLP(1280, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)` | `MINIMAL.md` L88: `MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)` | input 1280 (MobileNetV2's pooled features) instead of the ViT's 512; `proj_dim` 512 from `scripts/launch_inet10.py` |
| (V, N, K) layout of projections | `MINIMAL.md` L93: `.reshape(N, V, -1).transpose(0, 1)` | we concatenate views view-major and reshape to (V, N, K); same tensor layout; BatchNorm sees the same set of crops |
| `lr_schedule`: linear warm-up from 0.01 x lr over 1 epoch, then cosine | `MINIMAL.md` L157-L160 | final value lr / 1000 as in `README.md` L75-L76 (MINIMAL uses `eta_min=1e-3`, an absolute value); written as a per-step list so a resumed run uses identical values |
| AdamW, one parameter group, weight decay on all parameters | `MINIMAL.md` L154-L156 | lr 5e-4 and weight decay 5e-4 from `README.md` L70-L73 ("for ResNets"); no online probe group |
| no gradient clipping | `MINIMAL.md` text ("without having to resort to gradient clipping") | none |
| bf16 autocast | `README.md` L74, `MINIMAL.md` L167 | no `GradScaler` (bf16 does not need loss scaling; MINIMAL creates one anyway) |
| `DataAugmentationLeJEPA` | `README.md` L55-L64 (view table) | see section 3 |

## 3. Choices we had to make (no official MobileNetV2 or SAR recipe exists)

1. **All views are global, 8 views at 224 px.** Paper Section 5.1: "For non-ViT architectures (e.g., ResNet), set global_views = all_views"; its listing's comment: "if resnet: skip with a_emb=g_emb". `scripts/launch_inet10.py` (small timm models, which include CNNs): `n_views=8`, `multi_crop=false`, `resolution=224`. DINO's 2 + 8 multi-crop was not copied.
2. **Crop scale 0.3 to 1.0** (README global views). Checked on SAR with the XML boxes: 93.3% of crops keep at least half the vehicle.
3. **Saturation, hue, random grayscale left out**: no effect on a gray image (identical output).
4. **Solarization off**: SAR reason, same as A1 (it inverts the bright scatterers).
5. **Blur**: README gives only p 0.5; we use PIL Gaussian blur with sigma 0.1 to 2 px (MINIMAL's sigma range; same operation as A1).
6. **Backbone and features**: the torchvision MobileNetV2 `features` with global average pooling (1280 values), exactly the backbone A0 and A1 fine-tune. The ViT in MINIMAL has an extra linear layer to 512 values inside the "backbone" (`num_classes=512`); adding one would change the exported architecture, so the projector starts from 1280 instead.
7. **Projector output 512**: from the CNN sweep script. The paper's ablation (ViT-L) found smaller outputs slightly better (best 64); not tested here.
8. **Batch 28 images**: at 32, this process (10.9 GB reserved) plus the CUDA context and other programs' GPU memory exceed the card's 12 GB, and Windows silently pages GPU memory to system RAM (12.8 ms per image instead of 7.4). At 28: 9.65 GB reserved, 7.5 ms per image. The paper recommends 128 or more but states SIGReg's minibatch bias is "not a concern even for minibatches as small as 16". SIGReg's value scales with the batch (the statistic is multiplied by N), so lambda's effective weight also depends on it; the paper's lambda was not retuned.
9. **Epochs 100**: same as A1 (Nischal's decision); about 1.9 times A1's GPU time. A time-matched 55-epoch run is a later experiment.
10. **ImageNet initialization** instead of training from scratch (the study protocol).
11. **No online linear probe** during SSL (MINIMAL trains one with labels); SSL here never reads labels.
12. **Collapse diagnostics** (`collapse_diagnostics`): logged only; never part of the loss or the checkpoint choice.
13. **GPU memory cap** (`train_ssl.py`): PyTorch's allocator is limited to 11 GB, so a too-large batch fails with "out of memory" instead of Windows silently moving GPU memory into system RAM (observed during measurement: about 100 times slower).

## 4. Downstream code

`data.py`, `model.py`, `engine.py`, `evaluate.py`, `train_supervised.py`, `utils.py`, `config_finetune.json` and `ssl_data.py` are copies of the files in `Experiments/mobileNetv2_DINO` (themselves the A0 code plus the documented A1 changes). Changes here: file paths, help texts and docstrings; `run_name` `A2_lejepa_mobilenetv2`; `ssl_data.flat_args` reads a `lejepa` config section instead of `dino`. Training settings in `config_finetune.json` are identical to A0 and A1. `plots.py`: the DINO SSL charts were replaced by LeJEPA ones; the fine-tuning and evaluation charts are unchanged. `preview_views.py` was adapted to this folder's views.

## 5. Engineering changes that do not change the recipe

- **Last input steps on the GPU.** The data loader returns each view as a uint8 gray image (after all random transforms); `to_network_input` then copies it into 3 channels, divides by 255 and normalizes with the ImageNet mean/std on the GPU. These are the same operations as `Grayscale(3) -> ToTensor() -> Normalize()` in A0/A1; the smoke test compares both on the same views (largest difference 7e-7, float rounding). Reason: 8 float32 RGB views per image (154 MB per batch) made the first pilot data-bound.
- **4 data-loading workers and pinned memory** (A1 used 2 workers, unpinned, before the page file existed).
