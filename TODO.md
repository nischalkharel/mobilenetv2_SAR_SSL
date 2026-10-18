# To do after the current A1 run

Proposed work, not done yet. Started 2026-10-14 while the first full DINO run (`20261014-101106_A1_dino_ssl_mobilenetv2_seed0`) was training. Do not change code that a running job uses.

## 1. Fine-tuning accuracy across SSL epochs (Nischal's request)

Goal: see how the downstream accuracy changes with SSL training length, by fine-tuning the backbone saved at epochs 5, 10, 15, ..., 100 (20 checkpoints).

- [ ] `train_ssl.py`: add a setting, e.g. `"export_every": 5` in `config_ssl.json`, that exports the **teacher backbone** at every 5th epoch (`backbone_teacher_epoch005.pth`, ...). Backbone files only (about 9 MB each, about 180 MB in total). Do not keep every full `checkpoint.pth` (162 MB each).
- [ ] Decide whether resume must refuse a changed `export_every` (it does not change training, so probably allowed, but record it).
- [ ] A small script to fine-tune and score all 20 backbones on validation in one go, with one results table and a chart (validation accuracy against SSL epoch). About 3 min per fine-tune, so about 1 hour.
- [ ] Rerun the smoke test after the change.
- [ ] Then a new SSL run with the exports (about 7.5 h), or decide whether one is needed at all.

Decide before running (research decisions for Nischal):
- [ ] **Analysis or selection?** If the curve is only for analysis, keep the pre-set rule (epoch 100) for A1's reported result. If the best epoch is *chosen* on validation, then the validation labels become part of A1's tuning budget, which must be reported. A0 had no such choice, so say that in the comparison. The test set is never used for this choice.
- [ ] **Epoch 50 of a 100-epoch run is not a 50-epoch run.** With the cosine schedule, at epoch 50 the learning rate is still high and the teacher momentum has not reached 1. The curve shows "partway through a 100-epoch schedule". If a shorter schedule looks just as good, confirm it with a real run of that length.
- [ ] Repeat with more SSL seeds before drawing conclusions from small differences between epochs.

## 2. Smaller items

- [ ] The printed "max 8.32" next to marginal entropy is misleading: it is measured over 128 teacher outputs per batch, so its practical ceiling is ln(128) = 4.85. Fix the printout and the README note.
- [ ] Delete the empty folder `outputs/ssl/20261014-100704_A1_dino_ssl_mobilenetv2_seed0` (a Ctrl+C before training; no results). Waiting for Nischal's OK.

## 3. A2 (LeJEPA)

- [x] Main A2 implemented, smoke-tested and piloted (`Experiments/mobileNetv2_LeJEPA`, 2026-10-16).
- [ ] Full A2 SSL run (100 epochs, about 14 h; started 2026-10-16), then fine-tune from the final backbone, validation, and (when frozen) test.
- [ ] Accuracy vs SSL epoch for A2 from the 5-epoch exports (analysis only).
- [ ] Possible later A2 variants (one at a time, chosen on validation): projector output 64 (paper's ViT ablation), lambda, batch (gradient checkpointing for 64+).
- [ ] Separate controlled experiment: LeJEPA with DINO's crops (2 x 224 + 8 x 96, scales as A1) and 100 epochs.

## 4. Open from earlier sessions

- [ ] Local crop scale ablation: 0.05 (original DINO) against our 0.15, chosen on validation, if compute allows.
- [ ] Repeat seeds for A0 and A1 once the protocol is fixed.
