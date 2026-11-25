# FAIR-CSAR SL chip classification (22 classes): SSL on the trainval slices, fine-tuned and tested on chips

Mean per-class accuracy on test_clean, % (headline); per split and mean (sd). Then overall accuracy, then both on the full test set.

## Mean per-class accuracy, test_clean

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 21.9 / 28.0 / 24.8 | 24.9 (3.1) | 30.5 / 29.5 / 27.5 | 29.2 (1.5) | 40.6 / 39.3 / 38.8 | 39.6 (0.9) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.2 / 28.6 / 27.9 | 26.2 (3.5) | 29.7 / 30.6 / 31.3 | 30.5 (0.8) | 43.0 / 42.1 / 44.8 | 43.3 (1.3) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.9 / 29.6 / 28.3 | 27.9 (1.9) | 31.7 / 35.2 / 33.4 | 33.4 (1.8) | 42.5 / 39.8 / 44.8 | 42.4 (2.5) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 24.4 / 31.0 / 27.1 | 27.5 (3.3) | 33.1 / 35.5 / 32.9 | 33.8 (1.4) | 41.1 / 40.0 / 45.4 | 42.2 (2.9) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 27.6 / 29.5 / 24.8 | 27.3 (2.4) | 34.3 / 35.1 / 29.4 | 32.9 (3.1) | 44.2 / 43.4 / 43.3 | 43.6 (0.5) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 27.1 / 31.0 / 27.8 | 28.6 (2.1) | 35.1 / 34.4 / 32.6 | 34.0 (1.3) | 42.2 / 44.3 / 44.5 | 43.7 (1.3) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 26.7 / 32.7 / 27.0 | 28.8 (3.3) | 32.7 / 31.8 / 34.1 | 32.9 (1.1) | 39.0 / 41.1 / 44.4 | 41.5 (2.7) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | 26.3 / 31.0 / 28.0 | 28.5 (2.4) | 34.9 / 34.4 / 32.1 | 33.8 (1.5) | 42.5 / 41.9 / 46.5 | 43.7 (2.5) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | 22.7 / 27.0 / 25.3 | 25.0 (2.2) | 30.0 / 32.1 / 32.1 | 31.4 (1.2) | 39.0 / 40.3 / 43.0 | 40.8 (2.1) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.0 / 24.0 / 22.2 | 22.7 (1.1) | 28.9 / 27.1 / 27.0 | 27.7 (1.1) | 36.3 / 36.5 / 37.6 | 36.8 (0.7) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.2 / 28.2 / 26.3 | 26.6 (1.5) | 30.6 / 32.6 / 32.4 | 31.9 (1.1) | 42.0 / 38.0 / 42.7 | 40.9 (2.5) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 23.0 / 27.4 / 27.1 | 25.8 (2.5) | 27.0 / 30.0 / 30.5 | 29.2 (1.9) | 42.6 / 37.9 / 41.7 | 40.7 (2.5) |

## Overall accuracy, test_clean

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 43.6 / 44.1 / 42.2 | 43.3 (0.9) | 51.4 / 47.6 / 49.4 | 49.5 (1.9) | 58.2 / 58.8 / 60.7 | 59.2 (1.3) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 46.0 / 47.4 / 45.0 | 46.2 (1.2) | 53.3 / 51.6 / 53.0 | 52.6 (0.9) | 60.9 / 62.3 / 65.3 | 62.8 (2.3) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 47.4 / 47.1 / 44.5 | 46.3 (1.6) | 53.5 / 53.9 / 55.2 | 54.2 (0.9) | 60.6 / 61.3 / 64.4 | 62.1 (2.0) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 46.7 / 48.0 / 44.9 | 46.5 (1.5) | 55.6 / 53.3 / 54.5 | 54.5 (1.2) | 59.9 / 61.3 / 65.3 | 62.1 (2.8) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 49.7 / 48.9 / 45.3 | 47.9 (2.4) | 56.8 / 54.9 / 53.7 | 55.1 (1.6) | 61.3 / 63.2 / 64.1 | 62.9 (1.4) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 49.1 / 49.9 / 46.1 | 48.3 (2.0) | 57.5 / 53.2 / 55.7 | 55.5 (2.2) | 62.5 / 64.2 / 64.2 | 63.6 (1.0) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 47.8 / 48.3 / 44.7 | 46.9 (2.0) | 56.2 / 53.8 / 53.6 | 54.5 (1.4) | 58.9 / 63.0 / 64.6 | 62.2 (2.9) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | 51.9 / 50.2 / 45.9 | 49.4 (3.1) | 58.7 / 55.0 / 55.6 | 56.4 (2.0) | 63.1 / 64.0 / 66.2 | 64.5 (1.6) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | 47.7 / 49.5 / 43.2 | 46.8 (3.2) | 55.1 / 51.9 / 55.2 | 54.1 (1.9) | 60.3 / 63.9 / 63.9 | 62.7 (2.1) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 39.4 / 43.0 / 39.7 | 40.7 (2.0) | 47.1 / 46.2 / 47.2 | 46.9 (0.5) | 56.1 / 56.8 / 59.3 | 57.4 (1.7) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 46.0 / 48.1 / 44.8 | 46.3 (1.7) | 53.6 / 52.2 / 54.8 | 53.5 (1.3) | 61.8 / 60.9 / 63.9 | 62.2 (1.6) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 41.3 / 46.1 / 41.9 | 43.1 (2.7) | 47.5 / 48.0 / 49.6 | 48.4 (1.1) | 59.5 / 59.9 / 63.7 | 61.0 (2.3) |

## Mean per-class accuracy, full test

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 22.0 / 27.7 / 24.7 | 24.8 (2.9) | 30.0 / 29.4 / 27.7 | 29.1 (1.2) | 40.7 / 39.0 / 38.6 | 39.4 (1.1) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.4 / 28.6 / 28.0 | 26.3 (3.4) | 29.6 / 30.8 / 31.5 | 30.6 (1.0) | 43.1 / 41.6 / 45.0 | 43.2 (1.7) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.9 / 29.6 / 28.1 | 27.9 (1.9) | 31.5 / 35.1 / 33.2 | 33.3 (1.8) | 42.4 / 39.7 / 44.7 | 42.3 (2.5) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 24.5 / 31.1 / 26.9 | 27.5 (3.3) | 32.8 / 35.4 / 32.6 | 33.6 (1.6) | 41.1 / 39.9 / 45.2 | 42.1 (2.8) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 27.7 / 29.6 / 24.9 | 27.4 (2.4) | 34.5 / 35.5 / 29.4 | 33.1 (3.2) | 44.2 / 43.2 / 43.4 | 43.6 (0.5) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 27.3 / 31.1 / 27.4 | 28.6 (2.2) | 34.8 / 34.4 / 32.7 | 33.9 (1.1) | 42.3 / 44.1 / 44.5 | 43.6 (1.1) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 26.7 / 32.4 / 26.8 | 28.6 (3.2) | 32.5 / 31.7 / 33.9 | 32.7 (1.1) | 39.2 / 41.2 / 44.3 | 41.6 (2.5) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | 26.5 / 30.6 / 27.8 | 28.3 (2.1) | 34.9 / 34.2 / 32.3 | 33.8 (1.3) | 42.3 / 42.0 / 46.2 | 43.5 (2.3) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | 22.9 / 27.1 / 25.3 | 25.1 (2.1) | 30.2 / 31.7 / 32.2 | 31.4 (1.1) | 38.9 / 40.1 / 42.8 | 40.6 (2.0) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.2 / 23.9 / 22.0 | 22.7 (1.1) | 28.9 / 27.0 / 27.0 | 27.7 (1.1) | 36.3 / 36.6 / 37.4 | 36.7 (0.6) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.2 / 28.1 / 26.4 | 26.6 (1.4) | 30.6 / 32.4 / 32.1 | 31.7 (1.0) | 42.3 / 38.3 / 42.5 | 41.0 (2.3) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 23.3 / 27.4 / 27.8 | 26.2 (2.5) | 27.1 / 30.1 / 30.9 | 29.4 (2.0) | 42.2 / 37.9 / 41.3 | 40.5 (2.3) |

## Overall accuracy, full test

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 42.5 / 43.0 / 40.8 | 42.1 (1.2) | 50.3 / 46.8 / 48.1 | 48.4 (1.8) | 57.1 / 57.9 / 59.4 | 58.1 (1.1) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 45.0 / 46.3 / 43.6 | 45.0 (1.4) | 52.2 / 50.7 / 51.7 | 51.5 (0.8) | 59.6 / 61.0 / 64.2 | 61.6 (2.4) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 46.4 / 45.9 / 42.9 | 45.1 (1.9) | 52.3 / 52.8 / 53.6 | 52.9 (0.6) | 59.3 / 60.3 / 63.1 | 60.9 (1.9) |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 45.7 / 46.9 / 43.4 | 45.3 (1.8) | 54.3 / 52.3 / 53.1 | 53.2 (1.0) | 58.7 / 60.0 / 63.8 | 60.9 (2.6) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 48.6 / 47.6 / 43.8 | 46.7 (2.5) | 55.5 / 54.0 / 52.3 | 53.9 (1.6) | 60.1 / 62.1 / 62.9 | 61.7 (1.5) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 48.1 / 48.6 / 44.3 | 47.0 (2.3) | 56.2 / 52.2 / 54.2 | 54.2 (2.0) | 61.3 / 63.0 / 62.9 | 62.4 (1.0) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 46.7 / 47.1 / 43.0 | 45.6 (2.3) | 54.9 / 52.9 / 52.3 | 53.4 (1.4) | 57.9 / 61.7 / 63.2 | 60.9 (2.7) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | 50.7 / 48.7 / 44.3 | 47.9 (3.3) | 57.4 / 53.9 / 54.2 | 55.2 (2.0) | 61.5 / 62.8 / 64.9 | 63.1 (1.7) |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | 46.6 / 48.3 / 41.7 | 45.5 (3.4) | 54.0 / 50.7 / 53.8 | 52.8 (1.8) | 59.1 / 62.7 / 62.6 | 61.5 (2.1) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 38.6 / 42.1 / 38.3 | 39.7 (2.1) | 46.3 / 45.3 / 46.1 | 45.9 (0.5) | 55.0 / 55.7 / 58.0 | 56.2 (1.6) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 44.9 / 47.0 / 43.4 | 45.1 (1.8) | 52.3 / 51.0 / 53.1 | 52.1 (1.0) | 60.6 / 59.8 / 62.6 | 61.0 (1.5) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 40.5 / 45.3 / 41.0 | 42.3 (2.6) | 46.7 / 47.4 / 48.5 | 47.5 (0.9) | 58.2 / 58.8 / 62.3 | 59.8 (2.2) |

## Paired differences, mean per-class accuracy on test_clean (points): method minus its reference, same split and budget

References: no SSL for the DINO rows; the arm's DINO row for the object-only rows and their controls.

| method | reference | 1pct (s1 / s2 / s3) | mean | 3pct (s1 / s2 / s3) | mean | 10pct (s1 / s2 / s3) | mean |
|---|---|---|---|---|---|---|---|
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | +0.4 / +0.6 / +3.1 | +1.3 | -0.8 / +1.1 / +3.8 | +1.4 | +2.4 / +2.8 / +5.9 | +3.7 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | +3.6 / +1.0 / +0.4 | +1.7 | +2.0 / +4.7 / +2.1 | +2.9 | -0.5 / -2.3 / +0.1 | -0.9 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | +2.1 / +2.4 / -0.8 | +1.2 | +3.4 / +4.9 / +1.6 | +3.3 | -1.9 / -2.1 / +0.7 | -1.1 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | +5.7 / +1.5 / -0.1 | +2.4 | +3.8 / +5.6 / +1.9 | +3.8 | +3.6 / +4.1 / +4.4 | +4.0 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | -0.5 / +1.5 / +3.0 | +1.3 | +0.8 / -0.6 / +3.2 | +1.1 | -2.0 / +0.9 / +1.3 | +0.1 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | -0.9 / +3.2 / +2.3 | +1.5 | -1.5 / -3.3 / +4.7 | -0.0 | -5.2 / -2.3 / +1.1 | -2.1 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | -0.7 / +0.0 / +0.3 | -0.1 | -0.2 / -0.1 / -0.5 | -0.2 | +0.4 / -2.4 / +2.0 | -0.0 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | -3.7 / -4.0 / -2.7 | -3.5 | -4.9 / -2.3 / +0.0 | -2.4 | -3.5 / -1.7 / -3.5 | -2.9 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | +0.1 / -4.1 / -2.6 | -2.2 | -1.6 / -2.4 / -0.4 | -1.5 | -4.3 / -2.8 / -1.2 | -2.8 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | +3.2 / +4.2 / +4.1 | +3.9 | +1.7 / +5.5 / +5.4 | +4.2 | +5.7 / +1.5 / +5.1 | +4.1 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | +1.0 / +3.5 / +4.9 | +3.1 | -1.9 / +2.9 / +3.5 | +1.5 | +6.2 / +1.4 / +4.1 | +3.9 |

## Accuracy per major category, test_clean, 1 % labels (mean over splits)

| method | Aircraft | Ship | Bridge | Tank | Tower_Crane |
|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 22.2 | 29.7 | 26.6 | 82.3 | 14.2 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.6 | 32.6 | 31.1 | 87.0 | 12.0 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 24.0 | 32.8 | 28.5 | 85.9 | 18.9 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 24.4 | 33.6 | 27.6 | 85.9 | 15.2 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 23.6 | 35.1 | 31.4 | 87.2 | 23.2 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.7 | 35.3 | 29.2 | 87.8 | 21.0 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 25.0 | 34.1 | 26.7 | 86.2 | 15.9 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | 25.9 | 37.5 | 26.4 | 88.5 | 19.9 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | 23.4 | 32.9 | 23.8 | 89.4 | 13.9 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.5 | 28.0 | 26.1 | 76.8 | 8.8 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 23.9 | 33.6 | 24.8 | 85.8 | 17.0 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 23.0 | 30.6 | 27.5 | 80.2 | 10.7 |

## Per-class accuracy, test_clean, 1 % labels (mean over splits)

| method | Airbus_A220 | Airbus_A320 | Airbus_A330 | Airfreighter | Boeing737 | Boeing747 | Boeing767 | Boeing777 | Fokker-50 | Gulfstream | Helicopter | Other_Aircraft | Bulk_Carrier | Container_Ship | General_Cargo_Ship | Motion_Defocusing_Ship | Oil_Tanker | Other_Ship | Warship | Bridge | Tank | Tower_Crane |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 33 | 9 | 28 | 6 | 12 | 31 | 36 | 8 | 26 | 8 | 32 | 16 | 21 | 25 | 35 | 52 | 8 | 23 | 17 | 27 | 82 | 14 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 37 | 9 | 36 | 6 | 22 | 33 | 40 | 10 | 33 | 2 | 13 | 14 | 29 | 20 | 38 | 50 | 8 | 29 | 19 | 31 | 87 | 12 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 32 | 15 | 45 | 17 | 30 | 40 | 42 | 6 | 28 | 2 | 12 | 18 | 29 | 15 | 40 | 49 | 9 | 27 | 23 | 28 | 86 | 19 |
| SSL-FAIR-CSAR tiles / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 32 | 14 | 43 | 6 | 30 | 31 | 39 | 11 | 28 | 6 | 19 | 19 | 30 | 21 | 39 | 49 | 9 | 30 | 21 | 28 | 86 | 15 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 31 | 10 | 36 | 6 | 26 | 33 | 37 | 14 | 31 | 8 | 18 | 21 | 24 | 16 | 41 | 45 | 7 | 35 | 20 | 31 | 87 | 23 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 35 | 18 | 44 | 3 | 29 | 25 | 44 | 10 | 44 | 5 | 10 | 22 | 22 | 20 | 41 | 50 | 12 | 33 | 23 | 29 | 88 | 21 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 31 | 14 | 45 | 11 | 26 | 31 | 45 | 9 | 46 | 3 | 18 | 23 | 26 | 23 | 39 | 49 | 10 | 31 | 24 | 27 | 86 | 16 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only, teacher (SARMAE) regions | 37 | 21 | 38 | 8 | 27 | 33 | 40 | 9 | 39 | 3 | 17 | 21 | 17 | 20 | 46 | 52 | 12 | 34 | 16 | 26 | 89 | 20 |
| SSL-FAIR-CSAR tiles, 500 epochs / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / teacher-size random windows (control) | 42 | 4 | 38 | 11 | 16 | 27 | 33 | 5 | 13 | 3 | 17 | 20 | 23 | 22 | 42 | 50 | 7 | 24 | 24 | 24 | 89 | 14 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 40 | 7 | 23 | 3 | 21 | 23 | 35 | 14 | 26 | 3 | 15 | 18 | 22 | 20 | 34 | 37 | 11 | 25 | 15 | 26 | 77 | 9 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 33 | 14 | 36 | 8 | 24 | 25 | 42 | 9 | 31 | 9 | 18 | 18 | 25 | 21 | 38 | 43 | 9 | 33 | 20 | 25 | 86 | 17 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 36 | 8 | 37 | 6 | 25 | 25 | 38 | 8 | 39 | 6 | 14 | 16 | 33 | 26 | 36 | 49 | 11 | 25 | 16 | 27 | 80 | 11 |
