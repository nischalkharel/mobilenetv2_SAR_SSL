# SSL on SARDet-100K scenes (XA) or on the ATRNet pool (A1-A3, P4, XI), tested on ATRNet-STAR SOC-40 (official test accuracy, %)

| method | 1 % (s1 / s2 / s3) | mean (sd) | 3 % | mean (sd) | 10 % | mean (sd) |
|---|---|---|---|---|---|---|
| A0 ImageNet (no SSL) | 20.7 / 20.8 / 20.7 | 20.7 (0.1) | 42.8 / 43.0 / 44.5 | 43.4 (0.9) | 71.2 / 71.7 / 72.3 | 71.7 (0.6) |
| A1 DINO on ATRNet | 35.1 / 34.7 / 35.6 | 35.1 (0.5) | 56.3 / 54.8 / 56.7 | 55.9 (1.0) | 77.9 / 77.6 / 77.4 | 77.6 (0.3) |
| A2 LeJEPA on ATRNet | 36.1 / 36.0 / 36.6 | 36.2 (0.3) | 57.3 / 56.7 / 55.8 | 56.6 (0.8) | 74.3 / 76.4 / 75.7 | 75.5 (1.1) |
| A3 SparK on ATRNet | 24.6 / 24.4 / 24.9 | 24.7 (0.2) | 50.7 / 50.6 / 49.6 | 50.3 (0.6) | 74.3 / 74.5 / 74.9 | 74.6 (0.3) |
| P4 CondDINO on ATRNet | 39.3 / 41.0 / 41.9 | 40.8 (1.3) | 59.6 / 60.3 / 60.4 | 60.1 (0.4) | 78.4 / 78.9 / 78.6 | 78.6 (0.2) |
| P4 scrambled (control) | 34.8 / - / - | 34.8 (0.0) | 56.3 / - / - | 56.3 (0.0) | 77.4 / - / - | 77.4 (0.0) |
| P4 3-field (control) | 34.7 / - / - | 34.7 (0.0) | 56.0 / - / - | 56.0 (0.0) | 76.8 / - / - | 76.8 (0.0) |
| XA DINO on SARDet | 17.1 / 18.0 / 18.1 | 17.8 (0.5) | 41.4 / 41.8 / 42.3 | 41.8 (0.4) | 71.3 / 72.8 / 72.7 | 72.3 (0.8) |
| XA LeJEPA on SARDet | 12.3 / 13.4 / 11.1 | 12.3 (1.2) | 31.6 / 34.4 / 32.4 | 32.8 (1.4) | 64.0 / 65.4 / 64.2 | 64.6 (0.8) |
| XA SparK on SARDet | 21.5 / 23.3 / 23.2 | 22.7 (1.0) | 47.4 / 48.5 / 48.2 | 48.0 (0.6) | 74.0 / 73.9 / 73.9 | 73.9 (0.1) |
| XA RegionDINO object-only on SARDet | 27.6 / 27.3 / 26.6 | 27.2 (0.5) | 48.8 / 49.3 / 48.8 | 49.0 (0.3) | 73.5 / 73.8 / 73.8 | 73.7 (0.2) |
| XA RegionDINO object-only, random windows (control) | 20.3 / 21.3 / 21.2 | 20.9 (0.6) | 45.5 / 46.2 / 44.6 | 45.4 (0.8) | 72.6 / 73.0 / 72.6 | 72.7 (0.2) |

## Paired differences per split (test accuracy points): SARDet-SSL minus its reference, and minus A0

Reference = the same method's ATRNet-SSL row (A1/A2/A3) for DINO/LeJEPA/SparK; = XA DINO on SARDet for the XA RegionDINO rows; = A1 for XI DINO (suite recipe) and = XI DINO for the XI RegionDINO rows (same engine, same pool, only the object views differ).

| method | budget | vs reference (s1 / s2 / s3) | mean | vs A0 (s1 / s2 / s3) | mean |
|---|---|---|---|---|---|
| XA DINO on SARDet | 1pct | -18.0 / -16.6 / -17.5 | -17.4 | -3.6 / -2.8 / -2.6 | -3.0 |
| XA DINO on SARDet | 3pct | -14.9 / -12.9 / -14.4 | -14.1 | -1.4 / -1.1 / -2.2 | -1.6 |
| XA DINO on SARDet | 10pct | -6.6 / -4.8 / -4.7 | -5.4 | +0.1 / +1.2 / +0.4 | +0.5 |
| XA LeJEPA on SARDet | 1pct | -23.8 / -22.6 / -25.5 | -24.0 | -8.5 / -7.4 / -9.6 | -8.5 |
| XA LeJEPA on SARDet | 3pct | -25.7 / -22.3 / -23.4 | -23.8 | -11.2 / -8.6 / -12.1 | -10.6 |
| XA LeJEPA on SARDet | 10pct | -10.2 / -11.0 / -11.5 | -10.9 | -7.2 / -6.2 / -8.1 | -7.2 |
| XA SparK on SARDet | 1pct | -3.1 / -1.1 / -1.7 | -2.0 | +0.7 / +2.5 / +2.5 | +1.9 |
| XA SparK on SARDet | 3pct | -3.3 / -2.1 / -1.4 | -2.3 | +4.6 / +5.6 / +3.7 | +4.6 |
| XA SparK on SARDet | 10pct | -0.3 / -0.6 / -1.0 | -0.6 | +2.8 / +2.3 / +1.6 | +2.2 |
| XA RegionDINO object-only on SARDet | 1pct | +10.5 / +9.3 / +8.5 | +9.4 | +6.9 / +6.5 / +5.9 | +6.4 |
| XA RegionDINO object-only on SARDet | 3pct | +7.3 / +7.5 / +6.5 | +7.1 | +6.0 / +6.3 / +4.3 | +5.5 |
| XA RegionDINO object-only on SARDet | 10pct | +2.1 / +1.0 / +1.1 | +1.4 | +2.3 / +2.2 / +1.5 | +2.0 |
| XA RegionDINO object-only, random windows (control) | 1pct | +3.2 / +3.2 / +3.1 | +3.2 | -0.4 / +0.5 / +0.5 | +0.2 |
| XA RegionDINO object-only, random windows (control) | 3pct | +4.1 / +4.4 / +2.3 | +3.6 | +2.7 / +3.3 / +0.0 | +2.0 |
| XA RegionDINO object-only, random windows (control) | 10pct | +1.2 / +0.2 / -0.0 | +0.5 | +1.3 / +1.4 / +0.3 | +1.0 |

Duplicate cells (newest used): ('A0 ImageNet (no SSL)', '3pct', 1): 20261013-150506_A0_imagenet_mobilenetv2_seed0 vs 20261018-190525_A0_imagenet_mobilenetv2_seed0; ('A1 DINO on ATRNet', '3pct', 1): 20261015-113640_A1_dino_mobilenetv2_seed0 vs 20261019-032810_A1_dino_mobilenetv2_seed0
