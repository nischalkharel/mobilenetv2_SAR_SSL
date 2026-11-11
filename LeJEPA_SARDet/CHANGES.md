# LeJEPA_SARDet: what was brought in from the ATRNet-STAR A2 folder, and what changed

Source folder: `mobilenetv2_SAR_SSL/Experiments/mobileNetv2_LeJEPA/` (the ATRNet-STAR A2 experiment; its own
record of changes against the authors' code is `LEJEPA_CHANGES.md` there, and the untouched authors' files are
in its `reference/lejepa-main_original/`). Paper: Balestriero and LeCun, LeJEPA, arXiv 2511.08544 v3.
Licence of the authors' code: CC BY-NC 4.0 (`lejepa_lib/LICENSE`, copied from the reference folder).

## 1. File by file

| File here | Brought in from | Changed? | Why |
|---|---|---|---|
| `lejepa_lib/univariate/base.py`, `univariate/epps_pulley.py`, `multivariate/slicing.py` | authors' library via the old `lejepa_lib/` | **byte-identical** (SHA-256 08d4f115..., e6554ee4..., 86c0fe3a... = `reference/SOURCE.md`) | SIGReg is the authors' code |
| `lejepa_lib/__init__.py`, `univariate/__init__.py`, `multivariate/__init__.py` | old folder (already new there) | unchanged | the originals also import tests that need scipy |
| `lejepa_lib/LICENSE` | `reference/lejepa-main_original/LICENSE` | unchanged | licence must travel with the code |
| `lejepa_model.py` | old `lejepa_model.py` | `DataAugmentationLeJEPA`, `RandomGaussianBlur`, `RandomSolarization`, `to_network_input` **removed**; ImageNet weights now loaded by `shared.backbone.load_imagenet_mobilenet_v2()` instead of `paths.MOBILENETV2_IMAGENET_WEIGHTS`; `LeJEPANet`, `build_sigreg`, `lejepa_loss`, `collapse_diagnostics`, `lr_schedule` unchanged | views and preprocessing are the suite's shared recipe (`shared/ssl_data.py`), identical for every method |
| `train_ssl.py` | old `train_ssl.py` | the loop `train_one_epoch` and the "network, loss, optimizer, schedule" block became `LeJEPAMethod.train_epoch` / `build`; everything else (config merging, run folder, records, manifests, pixel cache, loader, checkpoint/resume, history, charts, export, GPU memory cap, view preview, `validate`) **dropped** because `shared/ssl_engine.py` does it | one engine for all methods; `validate()` survives as the checks in `LeJEPAMethod.__init__` |
| `config_ssl.json` | old `config_ssl.json` | only the method-specific keys kept (`lejepa_views`, `lejepa`, `optim`); `setup` (seed, csv, cache, workers, device, export_every) and the shared view / batch / epoch keys removed | those live in `shared/ssl_common.json` and may not be repeated (the engine raises) |
| `config_finetune.json`, `run_finetune.py` | `DINO_SARDet` | run name `lejepa_mobilenetv2` | the fine-tune stage is shared and identical for every method |
| `smoke_test.py` | `DINO_SARDet/smoke_test.py` structure + the LeJEPA checks of the old smoke test (sections 4 and 7) | LeJEPA-specific: SIGReg small for Gaussian / large for constant and squashed data, agreement of identical views 0, total = mix, lr schedule shape, export key names = torchvision `features`, SIGReg `global_step` == optimizer steps after a resume | same smoke structure as every other experiment folder |
| not brought in | `data.py`, `engine.py`, `evaluate.py`, `export_backbone.py`, `model.py`, `paths.py`, `plots.py`, `preview_views.py`, `ssl_data.py`, `train_supervised.py`, `utils.py` | | classification-only or replaced by `shared/` |

## 2. What changed in the METHOD (and what did not)

The method is the same as A2: one network (MobileNetV2 `features` + projector), 8 views per image all at the
global size, agreement toward the mean of the views, SIGReg on the projections, loss `lamb * SIGReg + (1 - lamb)
* agreement`, AdamW on all parameters, 1-epoch warm-up then cosine, bf16 autocast, no teacher, no EMA, no
stop-gradient, no centering, no gradient clipping. The exported backbone is the encoder of the FINAL epoch.

| Aspect | ATRNet-STAR A2 | Here | Why |
|---|---|---|---|
| Input images | 128 px chips, one object per chip | SARDet-100K images of 256 to 1000 px, first resized by the detector's rule (shorter side 512) | the shared view recipe (DESIGN.md section 4): object scale during SSL equals object scale during detection |
| View crop | `RandomResizedCrop(224, scale=(0.3, 1.0))` of the 128 px chip (upsampled) | square window of 224 x zoom, zoom in [0.75, 1.33], at the 512 scale, resized to 224 | shared recipe; the old scale range was tied to the 128 px chips and does not transfer to 512 px scenes |
| Flip, brightness/contrast 0.4 (p 0.8), blur sigma 0.1-2, no solarisation | same values | same values, from `shared/ssl_common.json` | unchanged (shared) |
| Blur probability | 0.5 for all views | 0.5 for all views (`lejepa_views.blur_p`, method-specific like DINO's) | unchanged |
| Preprocessing | uint8 gray -> RGB /255, ImageNet mean/std on the GPU | `shared.ssl_data.to_network_input`, the same arithmetic | unchanged |
| Batch, epochs, workers, seed | 64 / 100 / 4 / 0 in the method config | 64 / 100 / 8 / split-1 from `shared/ssl_common.json` | shared for every method; seed = split - 1 |
| Export every 5 epochs | yes (for the accuracy-versus-epoch analysis) | final epoch only | the engine exports once; the per-epoch `checkpoint.pth` still holds the encoder |
| Windows 11 GB allocator cap, pixel cache, view preview | yes | no | engine conveniences of the old folder, not part of the method |
| Learning-rate schedule values | per-step list, fingerprinted | same, fingerprint checked by the engine on resume | unchanged |

Equal epochs are not equal compute: one LeJEPA epoch forwards and back-propagates 8 x 224 px views per image,
about 1.8 to 1.9 x a DINO epoch on ATRNet-STAR; the engine records GPU-hours and view-forwards per run.

## 3. Exact hyperparameters (config_ssl.json + shared/ssl_common.json)

| Setting | Value | Where |
|---|---|---|
| Views | 8 global views, 224 px, no local views | `lejepa_views.n_global_views` = 8; `global_size` shared |
| Zoom jitter | [0.75, 1.33] of the view size at the 512 scale | shared |
| Horizontal flip | p 0.5 | shared |
| Brightness / contrast jitter | 0.4 / 0.4, p 0.8 | shared |
| Gaussian blur | p 0.5 (all views), sigma [0.1, 2.0] | `lejepa_views.blur_p`; sigma shared |
| Solarisation | off | shared |
| Backbone | torchvision MobileNetV2 `features` from the ImageNet-1k V2 file, global average pool -> 1280 | `lejepa_model.MobileNetV2Backbone` |
| Projector | MLP 1280 -> 2048 -> 2048 -> 512, BatchNorm1d + ReLU after each hidden layer | `lejepa.proj_hidden_dim` 2048, `lejepa.proj_dim` 512 |
| SIGReg | `SlicingUnivariateTest(EppsPulley(t_max=3.0, n_points=17), num_slices=1024)`; directions seeded from the `global_step` buffer (saved in the checkpoint) | `lejepa.t_max`, `n_points`, `num_slices` |
| Loss | `0.05 * SIGReg + 0.95 * agreement`, computed in float32 outside autocast | `lejepa.lamb` 0.05 |
| Optimizer | AdamW, lr 5e-4 (absolute, not batch-scaled), weight decay 5e-4 on ALL parameters, no clipping | `optim` |
| Schedule | per step: linear warm-up over 1 epoch from 0.01 x lr to lr, then cosine to 0.001 x lr | `optim.warmup_epochs` 1, `warmup_start_factor` 0.01, `final_lr_ratio` 0.001 |
| Precision | bf16 autocast for the network, no GradScaler; fp32 on CPU | `optim.precision` |
| Batch / epochs / seed | 64 images (512 views per step) / 100 / split - 1 | shared |
| Export | encoder `features.*` state dict (params + BN buffers) of the final epoch, torchvision key names | `LeJEPAMethod.export_state` |
| Smoke mode | 48 images, batch 8, 2 epochs, warm-up 1 epoch (`smoke_overrides`) | engine + `LeJEPAMethod.smoke_overrides` |

## 4. Checks in smoke_test.py specific to LeJEPA

- SIGReg of standard-normal (8, 256, 64) data is < 2 (about 1); of a constant and of a 0.01-scaled cloud it is
  more than 5 x larger.
- Agreement of 8 identical views is exactly 0; agreement equals MINIMAL.md's `(proj.mean(0) - proj).square().mean()`.
- `total == 0.05 * SIGReg + 0.95 * agreement`; SIGReg advances `global_step` by one per call.
- lr schedule: 0.01 x lr at step 0, lr at the end of warm-up, about 0.001 x lr at the last step.
- The exported state has exactly the key names and shapes of `torchvision.models.mobilenet_v2().features`.
- After stop-after-epoch-1 + resume, `checkpoint.pth` has SIGReg `global_step` == optimizer steps.
