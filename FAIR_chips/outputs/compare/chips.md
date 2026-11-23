# FAIR-CSAR SL chip classification (22 classes): SSL on the trainval slices, fine-tuned and tested on chips

Mean per-class accuracy on test_clean, % (headline); per split and mean (sd). Then overall accuracy, then both on the full test set.

## Mean per-class accuracy, test_clean

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 21.9 / 28.0 / 24.8 | 24.9 (3.1) | 30.5 / 29.5 / 27.5 | 29.2 (1.5) | 40.6 / 39.3 / 38.8 | 39.6 (0.9) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.0 / 24.0 / 22.2 | 22.7 (1.1) | 28.9 / 27.1 / 27.0 | 27.7 (1.1) | 36.3 / 36.5 / 37.6 | 36.8 (0.7) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.2 / 28.2 / 26.3 | 26.6 (1.5) | 30.6 / 32.6 / 32.4 | 31.9 (1.1) | 42.0 / 38.0 / 42.7 | 40.9 (2.5) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 23.0 / 27.4 / 27.1 | 25.8 (2.5) | 27.0 / 30.0 / 30.5 | 29.2 (1.9) | 42.6 / 37.9 / 41.7 | 40.7 (2.5) |

## Overall accuracy, test_clean

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 43.6 / 44.1 / 42.2 | 43.3 (0.9) | 51.4 / 47.6 / 49.4 | 49.5 (1.9) | 58.2 / 58.8 / 60.7 | 59.2 (1.3) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 39.4 / 43.0 / 39.7 | 40.7 (2.0) | 47.1 / 46.2 / 47.2 | 46.9 (0.5) | 56.1 / 56.8 / 59.3 | 57.4 (1.7) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 46.0 / 48.1 / 44.8 | 46.3 (1.7) | 53.6 / 52.2 / 54.8 | 53.5 (1.3) | 61.8 / 60.9 / 63.9 | 62.2 (1.6) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 41.3 / 46.1 / 41.9 | 43.1 (2.7) | 47.5 / 48.0 / 49.6 | 48.4 (1.1) | 59.5 / 59.9 / 63.7 | 61.0 (2.3) |

## Mean per-class accuracy, full test

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 22.0 / 27.7 / 24.7 | 24.8 (2.9) | 30.0 / 29.4 / 27.7 | 29.1 (1.2) | 40.7 / 39.0 / 38.6 | 39.4 (1.1) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.2 / 23.9 / 22.0 | 22.7 (1.1) | 28.9 / 27.0 / 27.0 | 27.7 (1.1) | 36.3 / 36.6 / 37.4 | 36.7 (0.6) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 25.2 / 28.1 / 26.4 | 26.6 (1.4) | 30.6 / 32.4 / 32.1 | 31.7 (1.0) | 42.3 / 38.3 / 42.5 | 41.0 (2.3) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 23.3 / 27.4 / 27.8 | 26.2 (2.5) | 27.1 / 30.1 / 30.9 | 29.4 (2.0) | 42.2 / 37.9 / 41.3 | 40.5 (2.3) |

## Overall accuracy, full test

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 42.5 / 43.0 / 40.8 | 42.1 (1.2) | 50.3 / 46.8 / 48.1 | 48.4 (1.8) | 57.1 / 57.9 / 59.4 | 58.1 (1.1) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 38.6 / 42.1 / 38.3 | 39.7 (2.1) | 46.3 / 45.3 / 46.1 | 45.9 (0.5) | 55.0 / 55.7 / 58.0 | 56.2 (1.6) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 44.9 / 47.0 / 43.4 | 45.1 (1.8) | 52.3 / 51.0 / 53.1 | 52.1 (1.0) | 60.6 / 59.8 / 62.6 | 61.0 (1.5) |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 40.5 / 45.3 / 41.0 | 42.3 (2.6) | 46.7 / 47.4 / 48.5 | 47.5 (0.9) | 58.2 / 58.8 / 62.3 | 59.8 (2.2) |

## Paired differences, mean per-class accuracy on test_clean (points): method minus its reference, same split and budget

References: no SSL for the DINO rows; the arm's DINO row for the object-only rows and their controls.

| method | reference | 1pct (s1 / s2 / s3) | mean | 3pct (s1 / s2 / s3) | mean | 10pct (s1 / s2 / s3) | mean |
|---|---|---|---|---|---|---|---|
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | +0.1 / -4.1 / -2.6 | -2.2 | -1.6 / -2.4 / -0.4 | -1.5 | -4.3 / -2.8 / -1.2 | -2.8 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | +3.2 / +4.2 / +4.1 | +3.9 | +1.7 / +5.5 / +5.4 | +4.2 | +5.7 / +1.5 / +5.1 | +4.1 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | +1.0 / +3.5 / +4.9 | +3.1 | -1.9 / +2.9 / +3.5 | +1.5 | +6.2 / +1.4 / +4.1 | +3.9 |

## Accuracy per major category, test_clean, 1 % labels (mean over splits)

| method | Aircraft | Ship | Bridge | Tank | Tower_Crane |
|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 22.2 | 29.7 | 26.6 | 82.3 | 14.2 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 22.5 | 28.0 | 26.1 | 76.8 | 8.8 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 23.9 | 33.6 | 24.8 | 85.8 | 17.0 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 23.0 | 30.6 | 27.5 | 80.2 | 10.7 |

## Per-class accuracy, test_clean, 1 % labels (mean over splits)

| method | Airbus_A220 | Airbus_A320 | Airbus_A330 | Airfreighter | Boeing737 | Boeing747 | Boeing767 | Boeing777 | Fokker-50 | Gulfstream | Helicopter | Other_Aircraft | Bulk_Carrier | Container_Ship | General_Cargo_Ship | Motion_Defocusing_Ship | Oil_Tanker | Other_Ship | Warship | Bridge | Tank | Tower_Crane |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSL-none / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / ImageNet start | 33 | 9 | 28 | 6 | 12 | 31 | 36 | 8 | 26 | 8 | 32 | 16 | 21 | 25 | 35 | 52 | 8 | 23 | 17 | 27 | 82 | 14 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / DINO | 40 | 7 | 23 | 3 | 21 | 23 | 35 | 14 | 26 | 3 | 15 | 18 | 22 | 20 | 34 | 37 | 11 | 25 | 15 | 26 | 77 | 9 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / RegionDINO object-only | 33 | 14 | 36 | 8 | 24 | 25 | 42 | 9 | 31 | 9 | 18 | 18 | 25 | 21 | 38 | 43 | 9 | 33 | 20 | 25 | 86 | 17 |
| SSL-SARDet-100K / FINETUNE-FAIR-CSAR / TEST-FAIR-CSAR / random windows (control) | 36 | 8 | 37 | 6 | 25 | 25 | 38 | 8 | 39 | 6 | 14 | 16 | 33 | 26 | 36 | 49 | 11 | 25 | 16 | 27 | 80 | 11 |
