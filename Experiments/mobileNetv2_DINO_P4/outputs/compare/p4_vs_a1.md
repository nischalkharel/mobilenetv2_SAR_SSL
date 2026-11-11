# P4 versus A1: CRC A100 runs, official SOC-40 test set (29,284 images), seed 0

Test accuracy (%), checkpoint chosen on validation. Differences in parentheses are against A1 on the same
split and budget. Test scores are recorded, never used for decisions. Seed noise is about 1 point.

## Test accuracy

**Split 1**

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 20.7 (-14.4) | 42.8 (-13.5) | 71.2 (-6.7) |
| DINO (A1) | 35.1 | 56.3 | 77.9 |
| DINO + per-cell centering (P4) | 39.3 (+4.2) | 59.6 (+3.3) | 78.4 (+0.5) |
| scrambled cells (P4perm) | 34.8 (-0.3) | 56.3 (-0.0) | 77.4 (-0.5) |
| 3-field cells (P4d) | 34.7 (-0.4) | 56.0 (-0.4) | 76.8 (-1.1) |

**Split 2**

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 20.8 (-13.9) | 43.0 (-11.8) | 71.7 (-6.0) |
| DINO (A1) | 34.7 | 54.8 | 77.6 |
| DINO + per-cell centering (P4) | 41.0 (+6.3) | 60.3 (+5.5) | 78.9 (+1.2) |

**Split 3**

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 20.7 (-14.9) | 44.5 (-12.2) | 72.3 (-5.0) |
| DINO (A1) | 35.6 | 56.7 | 77.4 |
| DINO + per-cell centering (P4) | 41.9 (+6.3) | 60.4 (+3.7) | 78.6 (+1.3) |

**Mean +/- standard deviation over splits 1-3** (methods run on all three splits)

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 20.7 +/- 0.1 (-14.4) | 43.4 +/- 0.9 (-12.5) | 71.7 +/- 0.6 (-5.9) |
| DINO (A1) | 35.1 +/- 0.5 | 55.9 +/- 1.0 | 77.6 +/- 0.3 |
| DINO + per-cell centering (P4) | 40.8 +/- 1.3 (+5.6) | 60.1 +/- 0.4 (+4.2) | 78.6 +/- 0.2 (+1.0) |

## Validation accuracy (800 images, the model-selection set)

**Split 1**

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 21.6 (-13.0) | 42.0 (-16.2) | 71.7 (-4.0) |
| DINO (A1) | 34.6 | 58.2 | 75.7 |
| DINO + per-cell centering (P4) | 40.0 (+5.4) | 59.5 (+1.3) | 78.4 (+2.6) |
| scrambled cells (P4perm) | 35.5 (+0.9) | 56.7 (-1.5) | 77.9 (+2.1) |
| 3-field cells (P4d) | 35.7 (+1.1) | 57.5 (-0.7) | 76.4 (+0.6) |

**Split 2**

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 23.3 (-10.1) | 45.0 (-10.0) | 71.6 (-4.4) |
| DINO (A1) | 33.4 | 55.0 | 76.0 |
| DINO + per-cell centering (P4) | 42.8 (+9.4) | 59.6 (+4.6) | 79.0 (+3.0) |

**Split 3**

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 20.9 (-15.8) | 42.8 (-13.6) | 69.7 (-4.8) |
| DINO (A1) | 36.6 | 56.4 | 74.5 |
| DINO + per-cell centering (P4) | 43.4 (+6.7) | 58.9 (+2.5) | 76.4 (+1.9) |

**Mean +/- standard deviation over splits 1-3** (methods run on all three splits)

| Method | 1 % labels | 3 % labels | 10 % labels |
|---|---|---|---|
| ImageNet only (A0) | 21.9 +/- 1.2 (-13.0) | 43.2 +/- 1.6 (-13.3) | 71.0 +/- 1.1 (-4.4) |
| DINO (A1) | 34.9 +/- 1.6 | 56.5 +/- 1.6 | 75.4 +/- 0.8 |
| DINO + per-cell centering (P4) | 42.0 +/- 1.8 (+7.2) | 59.3 +/- 0.4 (+2.8) | 77.9 +/- 1.4 (+2.5) |

- DINO + per-cell centering (P4) minus A1, test accuracy per split: 1 %: +4.2, +6.3, +6.3; 3 %: +3.3, +5.5, +3.7; 10 %: +0.5, +1.2, +1.3
- scrambled cells (P4perm) minus A1, test accuracy per split: 1 %: -0.3; 3 %: -0.0; 10 %: -0.5
- 3-field cells (P4d) minus A1, test accuracy per split: 1 %: -0.4; 3 %: -0.4; 10 %: -1.1

## SSL cost on one A100 (100 epochs)

| Method | Split | Hours | Peak GPU GB | Cells | SSL run |
|---|---|---|---|---|---|
| DINO (A1) | 1 | 7.8 | 9.2 | - | 20261018-192425_A1_dino_ssl_mobilenetv2_seed0 |
| DINO (A1) | 2 | 7.8 | 9.2 | - | 20261028-170414_A1_dino_ssl_mobilenetv2_split2_seed0 |
| DINO (A1) | 3 | 7.8 | 9.2 | - | 20261028-190646_A1_dino_ssl_mobilenetv2_split3_seed0 |
| DINO + per-cell centering (P4) | 1 | 8.0 | 9.2 | 432 | 20261110-194252_P4_dino_ssl_mobilenetv2_seed0 |
| DINO + per-cell centering (P4) | 2 | 8.0 | 9.2 | 432 | 20261110-150758_P4_dino_ssl_mobilenetv2_split2_seed0 |
| DINO + per-cell centering (P4) | 3 | 8.0 | 9.2 | 432 | 20261110-150731_P4_dino_ssl_mobilenetv2_split3_seed0 |
| scrambled cells (P4perm) | 1 | 7.9 | 9.2 | 432 | 20261110-204010_P4perm_dino_ssl_mobilenetv2_seed0 |
| 3-field cells (P4d) | 1 | 9.0 | 9.2 | 36 | 20261110-150743_P4d_dino_ssl_mobilenetv2_seed0 |

Runs: 51 fine-tunes with test evaluation. Written 2026-11-11T09:29:45.
