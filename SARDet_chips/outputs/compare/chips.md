# SARDet-100K chip classification (study S1: SSL on the scenes, fine-tuned and tested on chips)

Mean per-class accuracy on test_clean, % (headline); per split and mean (sd). Then overall accuracy on test_clean, then both on the full test set.


## Mean per-class accuracy, test_clean

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 82.9 / 80.5 / 79.6 | 81.0 (1.7) | 92.5 / 91.5 / 90.2 | 91.4 (1.2) | 96.3 / 97.6 / 95.6 | 96.5 (1.0) |
| DINO on scenes | 86.4 / 81.5 / 78.8 | 82.2 (3.9) | 92.8 / 93.3 / 90.2 | 92.1 (1.6) | 96.5 / 97.9 / 96.4 | 97.0 (0.8) |
| LeJEPA on scenes | 82.5 / 82.3 / 79.1 | 81.3 (1.9) | 92.5 / 91.1 / 89.2 | 90.9 (1.7) | 95.6 / 97.7 / 96.2 | 96.5 (1.1) |
| SparK on scenes | 84.0 / 83.9 / 79.1 | 82.3 (2.8) | 91.6 / 90.7 / 89.4 | 90.6 (1.1) | 95.8 / 97.7 / 96.5 | 96.7 (0.9) |
| RegionDINO scene+object | 85.3 / 84.6 / 77.2 | 82.4 (4.5) | 92.8 / 91.6 / 90.0 | 91.5 (1.4) | 96.9 / 98.3 / 96.1 | 97.1 (1.1) |
| RegionDINO scene+object, random windows (control) | 86.0 / 85.3 / 79.6 | 83.6 (3.5) | 91.3 / 92.1 / 90.4 | 91.3 (0.9) | 96.8 / 98.1 / 96.8 | 97.2 (0.8) |
| RegionDINO object-only | 85.4 / 82.6 / 79.3 | 82.4 (3.0) | 92.7 / 92.3 / 90.6 | 91.9 (1.1) | 96.3 / 97.0 / 97.0 | 96.8 (0.4) |
| RegionDINO object-only, random windows (control) | 82.8 / 80.2 / 78.5 | 80.5 (2.2) | 91.2 / 94.8 / 90.3 | 92.1 (2.4) | 96.9 / 98.3 / 96.5 | 97.3 (1.0) |
| RegionDINO object-only, teacher regions | 84.7 / 81.2 / 80.2 | 82.0 (2.3) | 92.8 / 92.6 / 88.8 | 91.4 (2.3) | 97.0 / 98.0 / 96.5 | 97.2 (0.8) |
| RegionDINO object-only, teacher-size random windows (control) | 84.7 / 79.2 / 77.3 | 80.4 (3.8) | 92.0 / 89.6 / 89.1 | 90.2 (1.6) | 96.0 / 97.9 / 96.6 | 96.8 (1.0) |
| SSL on FAIR-CSAR tiles: DINO | 83.2 / 81.2 / 78.9 | 81.1 (2.2) | 93.7 / 93.0 / 91.6 | 92.8 (1.1) | 96.5 / 97.9 / 97.3 | 97.2 (0.7) |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 84.5 / 80.4 / 82.5 | 82.5 (2.0) | 91.0 / 91.1 / 89.8 | 90.6 (0.7) | 96.4 / 97.3 / 96.5 | 96.7 (0.5) |
| SSL on FAIR-CSAR tiles: random windows (control) | 84.1 / 83.0 / 78.8 | 81.9 (2.8) | 92.1 / 92.2 / 90.4 | 91.6 (1.0) | 95.6 / 98.3 / 96.3 | 96.7 (1.4) |
| SSL on FAIR-CSAR tiles: teacher regions | 83.0 / 80.7 / 79.3 | 81.0 (1.9) | 91.5 / 91.7 / 89.8 | 91.0 (1.0) | 96.3 / 97.5 / 97.2 | 97.0 (0.6) |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 83.5 / 80.9 / 80.0 | 81.5 (1.9) | 92.3 / 91.7 / 91.1 | 91.7 (0.6) | 95.9 / 97.9 / 96.6 | 96.8 (1.0) |

## Overall accuracy, test_clean

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 92.2 / 90.6 / 92.2 | 91.7 (1.0) | 96.9 / 96.8 / 96.2 | 96.7 (0.4) | 98.7 / 98.9 / 97.7 | 98.4 (0.6) |
| DINO on scenes | 93.8 / 91.8 / 91.1 | 92.3 (1.4) | 97.5 / 96.8 / 95.7 | 96.7 (0.9) | 98.9 / 99.1 / 99.0 | 99.0 (0.1) |
| LeJEPA on scenes | 93.2 / 91.9 / 90.9 | 92.0 (1.1) | 97.4 / 96.5 / 95.6 | 96.5 (0.9) | 98.9 / 99.1 / 98.8 | 98.9 (0.1) |
| SparK on scenes | 93.1 / 92.1 / 92.4 | 92.5 (0.5) | 97.0 / 96.1 / 96.0 | 96.4 (0.6) | 98.7 / 99.0 / 98.5 | 98.7 (0.3) |
| RegionDINO scene+object | 93.6 / 93.5 / 91.1 | 92.7 (1.4) | 97.0 / 96.8 / 95.5 | 96.4 (0.8) | 99.1 / 99.3 / 98.3 | 98.9 (0.5) |
| RegionDINO scene+object, random windows (control) | 93.4 / 92.3 / 91.0 | 92.2 (1.2) | 96.9 / 96.7 / 95.6 | 96.4 (0.7) | 99.0 / 99.2 / 99.0 | 99.1 (0.1) |
| RegionDINO object-only | 94.3 / 92.4 / 91.1 | 92.6 (1.6) | 97.3 / 96.8 / 96.4 | 96.8 (0.5) | 98.6 / 99.0 / 98.7 | 98.8 (0.2) |
| RegionDINO object-only, random windows (control) | 92.5 / 91.5 / 91.7 | 91.9 (0.6) | 97.1 / 96.8 / 96.4 | 96.8 (0.4) | 99.0 / 99.3 / 98.8 | 99.0 (0.2) |
| RegionDINO object-only, teacher regions | 93.1 / 91.9 / 90.9 | 92.0 (1.1) | 97.3 / 96.8 / 95.7 | 96.6 (0.8) | 99.0 / 99.3 / 98.6 | 99.0 (0.3) |
| RegionDINO object-only, teacher-size random windows (control) | 93.3 / 91.4 / 91.4 | 92.0 (1.1) | 96.9 / 95.8 / 95.8 | 96.2 (0.6) | 98.7 / 99.2 / 99.0 | 98.9 (0.3) |
| SSL on FAIR-CSAR tiles: DINO | 92.6 / 92.2 / 91.6 | 92.1 (0.5) | 97.1 / 96.9 / 95.7 | 96.6 (0.7) | 98.7 / 99.0 / 99.0 | 98.9 (0.1) |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 93.3 / 91.3 / 93.1 | 92.6 (1.1) | 95.0 / 95.9 / 96.2 | 95.7 (0.6) | 98.8 / 99.0 / 98.8 | 98.9 (0.1) |
| SSL on FAIR-CSAR tiles: random windows (control) | 93.7 / 92.0 / 91.9 | 92.6 (1.0) | 96.7 / 96.7 / 96.1 | 96.5 (0.3) | 98.7 / 99.2 / 98.2 | 98.7 (0.5) |
| SSL on FAIR-CSAR tiles: teacher regions | 92.3 / 91.7 / 91.4 | 91.8 (0.5) | 96.6 / 96.7 / 96.1 | 96.5 (0.3) | 98.8 / 98.9 / 98.7 | 98.8 (0.1) |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 91.0 / 91.7 / 92.2 | 91.6 (0.6) | 96.7 / 96.6 / 96.7 | 96.7 (0.0) | 98.6 / 99.2 / 98.7 | 98.8 (0.3) |

## Mean per-class accuracy, full test

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 82.8 / 80.3 / 79.5 | 80.8 (1.7) | 92.2 / 91.2 / 90.1 | 91.2 (1.1) | 96.0 / 97.5 / 95.6 | 96.4 (1.0) |
| DINO on scenes | 86.2 / 81.1 / 78.7 | 82.0 (3.8) | 92.5 / 93.0 / 90.1 | 91.9 (1.5) | 96.2 / 97.9 / 96.3 | 96.8 (0.9) |
| LeJEPA on scenes | 82.2 / 81.8 / 79.0 | 81.0 (1.8) | 92.2 / 90.6 / 88.8 | 90.5 (1.7) | 95.4 / 97.6 / 96.0 | 96.3 (1.1) |
| SparK on scenes | 83.8 / 83.8 / 79.0 | 82.2 (2.7) | 91.3 / 90.5 / 89.1 | 90.3 (1.1) | 95.6 / 97.7 / 96.5 | 96.6 (1.0) |
| RegionDINO scene+object | 85.2 / 84.2 / 77.0 | 82.1 (4.5) | 92.5 / 91.4 / 89.9 | 91.3 (1.3) | 96.8 / 98.2 / 95.9 | 97.0 (1.1) |
| RegionDINO scene+object, random windows (control) | 85.9 / 85.0 / 79.4 | 83.4 (3.5) | 91.1 / 91.9 / 90.2 | 91.1 (0.8) | 96.5 / 98.1 / 96.6 | 97.1 (0.9) |
| RegionDINO object-only | 85.3 / 82.3 / 79.3 | 82.3 (3.0) | 92.4 / 92.1 / 90.6 | 91.7 (1.0) | 96.2 / 97.0 / 96.9 | 96.7 (0.5) |
| RegionDINO object-only, random windows (control) | 82.7 / 79.9 / 78.4 | 80.3 (2.2) | 91.0 / 94.7 / 90.3 | 92.0 (2.4) | 96.7 / 98.3 / 96.4 | 97.1 (1.0) |
| RegionDINO object-only, teacher regions | 84.5 / 80.9 / 80.1 | 81.8 (2.4) | 92.5 / 92.4 / 88.6 | 91.2 (2.2) | 96.7 / 98.0 / 96.5 | 97.1 (0.8) |
| RegionDINO object-only, teacher-size random windows (control) | 84.6 / 78.7 / 77.3 | 80.2 (3.9) | 91.8 / 89.3 / 89.0 | 90.0 (1.5) | 95.7 / 97.8 / 96.5 | 96.7 (1.1) |
| SSL on FAIR-CSAR tiles: DINO | 83.1 / 80.9 / 78.8 | 80.9 (2.2) | 93.5 / 92.9 / 91.5 | 92.6 (1.0) | 96.3 / 97.9 / 97.2 | 97.1 (0.8) |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 84.3 / 80.1 / 82.5 | 82.3 (2.1) | 90.7 / 90.8 / 89.7 | 90.4 (0.6) | 96.2 / 97.2 / 96.4 | 96.6 (0.6) |
| SSL on FAIR-CSAR tiles: random windows (control) | 83.9 / 82.7 / 78.8 | 81.8 (2.7) | 91.9 / 91.8 / 90.4 | 91.4 (0.8) | 95.5 / 98.2 / 96.2 | 96.7 (1.4) |
| SSL on FAIR-CSAR tiles: teacher regions | 82.8 / 80.4 / 79.2 | 80.8 (1.9) | 91.2 / 91.5 / 89.8 | 90.8 (0.9) | 96.2 / 97.5 / 97.2 | 97.0 (0.7) |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 83.4 / 80.4 / 80.0 | 81.2 (1.9) | 92.0 / 91.3 / 91.0 | 91.5 (0.5) | 95.8 / 97.9 / 96.5 | 96.7 (1.1) |

## Overall accuracy, full test

| method | 1pct (s1 / s2 / s3) | mean (sd) | 3pct (s1 / s2 / s3) | mean (sd) | 10pct (s1 / s2 / s3) | mean (sd) |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 92.1 / 90.4 / 92.0 | 91.5 (1.0) | 96.8 / 96.7 / 96.2 | 96.5 (0.3) | 98.6 / 98.8 / 97.7 | 98.4 (0.6) |
| DINO on scenes | 93.7 / 91.6 / 90.9 | 92.1 (1.5) | 97.4 / 96.7 / 95.6 | 96.6 (0.9) | 98.8 / 99.1 / 98.9 | 98.9 (0.1) |
| LeJEPA on scenes | 92.9 / 91.7 / 90.8 | 91.8 (1.1) | 97.3 / 96.3 / 95.4 | 96.3 (0.9) | 98.8 / 99.0 / 98.8 | 98.9 (0.1) |
| SparK on scenes | 93.0 / 91.9 / 92.2 | 92.4 (0.6) | 96.9 / 96.0 / 95.9 | 96.3 (0.6) | 98.7 / 99.0 / 98.4 | 98.7 (0.3) |
| RegionDINO scene+object | 93.4 / 93.2 / 90.9 | 92.5 (1.4) | 96.9 / 96.6 / 95.5 | 96.3 (0.7) | 99.0 / 99.2 / 98.3 | 98.8 (0.5) |
| RegionDINO scene+object, random windows (control) | 93.3 / 92.2 / 90.8 | 92.1 (1.3) | 96.8 / 96.5 / 95.5 | 96.3 (0.7) | 99.0 / 99.2 / 98.9 | 99.0 (0.1) |
| RegionDINO object-only | 94.2 / 92.2 / 91.0 | 92.5 (1.6) | 97.2 / 96.6 / 96.3 | 96.7 (0.5) | 98.6 / 98.9 / 98.7 | 98.7 (0.2) |
| RegionDINO object-only, random windows (control) | 92.4 / 91.3 / 91.5 | 91.7 (0.6) | 97.0 / 96.7 / 96.4 | 96.7 (0.3) | 99.0 / 99.2 / 98.8 | 99.0 (0.2) |
| RegionDINO object-only, teacher regions | 92.9 / 91.7 / 90.7 | 91.8 (1.1) | 97.2 / 96.6 / 95.6 | 96.5 (0.8) | 98.9 / 99.3 / 98.6 | 98.9 (0.3) |
| RegionDINO object-only, teacher-size random windows (control) | 93.1 / 91.2 / 91.2 | 91.8 (1.1) | 96.8 / 95.6 / 95.7 | 96.0 (0.7) | 98.6 / 99.1 / 98.9 | 98.9 (0.3) |
| SSL on FAIR-CSAR tiles: DINO | 92.5 / 91.9 / 91.4 | 91.9 (0.5) | 97.0 / 96.8 / 95.7 | 96.5 (0.7) | 98.7 / 98.9 / 99.0 | 98.9 (0.1) |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 93.2 / 91.0 / 92.9 | 92.4 (1.2) | 94.9 / 95.7 / 96.1 | 95.6 (0.6) | 98.8 / 98.9 / 98.8 | 98.8 (0.1) |
| SSL on FAIR-CSAR tiles: random windows (control) | 93.6 / 91.8 / 91.7 | 92.4 (1.1) | 96.6 / 96.5 / 96.0 | 96.4 (0.3) | 98.6 / 99.2 / 98.2 | 98.7 (0.5) |
| SSL on FAIR-CSAR tiles: teacher regions | 92.2 / 91.5 / 91.2 | 91.6 (0.5) | 96.4 / 96.5 / 96.1 | 96.3 (0.3) | 98.7 / 98.9 / 98.6 | 98.7 (0.1) |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 90.8 / 91.5 / 92.0 | 91.4 (0.6) | 96.6 / 96.5 / 96.6 | 96.6 (0.1) | 98.5 / 99.2 / 98.7 | 98.8 (0.3) |

## Paired differences, mean per-class accuracy on test_clean (points): method minus its reference, same split and budget

Reference: no SSL for DINO / LeJEPA / SparK; plain DINO for the RegionDINO rows and controls.

| method | budget | vs reference (s1 / s2 / s3) | mean | vs no SSL | mean |
|---|---|---|---|---|---|
| DINO on scenes | 1pct | +3.5 / +1.1 / -0.8 | +1.3 | +3.5 / +1.1 / -0.8 | +1.3 |
| DINO on scenes | 3pct | +0.3 / +1.8 / -0.0 | +0.7 | +0.3 / +1.8 / -0.0 | +0.7 |
| DINO on scenes | 10pct | +0.3 / +0.3 / +0.8 | +0.5 | +0.3 / +0.3 / +0.8 | +0.5 |
| LeJEPA on scenes | 1pct | -0.4 / +1.8 / -0.5 | +0.3 | -0.4 / +1.8 / -0.5 | +0.3 |
| LeJEPA on scenes | 3pct | +0.0 / -0.4 / -1.1 | -0.5 | +0.0 / -0.4 / -1.1 | -0.5 |
| LeJEPA on scenes | 10pct | -0.7 / +0.1 / +0.5 | -0.0 | -0.7 / +0.1 / +0.5 | -0.0 |
| SparK on scenes | 1pct | +1.1 / +3.4 / -0.4 | +1.4 | +1.1 / +3.4 / -0.4 | +1.4 |
| SparK on scenes | 3pct | -1.0 / -0.8 / -0.9 | -0.9 | -1.0 / -0.8 / -0.9 | -0.9 |
| SparK on scenes | 10pct | -0.4 / +0.1 / +0.9 | +0.2 | -0.4 / +0.1 / +0.9 | +0.2 |
| RegionDINO scene+object | 1pct | -1.1 / +3.1 / -1.6 | +0.1 | +2.4 / +4.1 / -2.4 | +1.4 |
| RegionDINO scene+object | 3pct | -0.0 / -1.7 / -0.2 | -0.6 | +0.3 / +0.1 / -0.2 | +0.0 |
| RegionDINO scene+object | 10pct | +0.4 / +0.3 / -0.4 | +0.1 | +0.7 / +0.6 / +0.4 | +0.6 |
| RegionDINO scene+object, random windows (control) | 1pct | -0.4 / +3.7 / +0.8 | +1.4 | +3.1 / +4.8 / +0.1 | +2.7 |
| RegionDINO scene+object, random windows (control) | 3pct | -1.5 / -1.1 / +0.1 | -0.8 | -1.2 / +0.6 / +0.1 | -0.2 |
| RegionDINO scene+object, random windows (control) | 10pct | +0.2 / +0.2 / +0.4 | +0.3 | +0.5 / +0.5 / +1.2 | +0.7 |
| RegionDINO object-only | 1pct | -1.0 / +1.0 / +0.6 | +0.2 | +2.5 / +2.1 / -0.2 | +1.5 |
| RegionDINO object-only | 3pct | -0.2 / -0.9 / +0.4 | -0.2 | +0.1 / +0.8 / +0.4 | +0.5 |
| RegionDINO object-only | 10pct | -0.2 / -0.9 / +0.6 | -0.2 | +0.1 / -0.6 / +1.4 | +0.3 |
| RegionDINO object-only, random windows (control) | 1pct | -3.5 / -1.4 / -0.3 | -1.7 | -0.0 / -0.3 / -1.1 | -0.5 |
| RegionDINO object-only, random windows (control) | 3pct | -1.6 / +1.6 / +0.0 | +0.0 | -1.3 / +3.3 / +0.0 | +0.7 |
| RegionDINO object-only, random windows (control) | 10pct | +0.4 / +0.4 / +0.1 | +0.3 | +0.7 / +0.7 / +0.9 | +0.7 |
| RegionDINO object-only, teacher regions | 1pct | -0.7 / -1.3 / +0.8 | -0.4 | +1.8 / +0.8 / +0.6 | +1.1 |
| RegionDINO object-only, teacher regions | 3pct | +0.1 / +0.3 / -1.9 | -0.5 | +0.3 / +1.1 / -1.5 | -0.0 |
| RegionDINO object-only, teacher regions | 10pct | +0.7 / +1.0 / -0.5 | +0.4 | +0.7 / +0.4 / +0.9 | +0.7 |
| RegionDINO object-only, teacher-size random windows (control) | 1pct | +0.0 / -2.1 / -2.9 | -1.6 | +1.8 / -1.3 / -2.2 | -0.6 |
| RegionDINO object-only, teacher-size random windows (control) | 3pct | -0.8 / -3.0 / +0.3 | -1.1 | -0.5 / -1.9 / -1.1 | -1.2 |
| RegionDINO object-only, teacher-size random windows (control) | 10pct | -1.0 / -0.2 / +0.0 | -0.4 | -0.3 / +0.2 / +0.9 | +0.3 |
| SSL on FAIR-CSAR tiles: DINO | 1pct | +0.4 / +0.7 / -0.7 | +0.1 | +0.4 / +0.7 / -0.7 | +0.1 |
| SSL on FAIR-CSAR tiles: DINO | 3pct | +1.2 / +1.5 / +1.4 | +1.4 | +1.2 / +1.5 / +1.4 | +1.4 |
| SSL on FAIR-CSAR tiles: DINO | 10pct | +0.2 / +0.3 / +1.7 | +0.7 | +0.2 / +0.3 / +1.7 | +0.7 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 1pct | +1.3 / -0.8 / +3.6 | +1.4 | +1.6 / -0.1 / +3.0 | +1.5 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 3pct | -2.7 / -1.9 / -1.8 | -2.2 | -1.5 / -0.4 / -0.4 | -0.8 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 10pct | -0.1 / -0.6 / -0.8 | -0.5 | +0.1 / -0.3 / +0.9 | +0.2 |
| SSL on FAIR-CSAR tiles: random windows (control) | 1pct | +0.8 / +1.8 / -0.1 | +0.8 | +1.2 / +2.5 / -0.8 | +1.0 |
| SSL on FAIR-CSAR tiles: random windows (control) | 3pct | -1.7 / -0.9 / -1.2 | -1.2 | -0.5 / +0.7 / +0.2 | +0.1 |
| SSL on FAIR-CSAR tiles: random windows (control) | 10pct | -0.9 / +0.4 / -1.0 | -0.5 | -0.7 / +0.6 / +0.7 | +0.2 |
| SSL on FAIR-CSAR tiles: teacher regions | 1pct | -1.5 / +0.3 / -3.2 | -1.5 | +0.1 / +0.2 / -0.3 | +0.0 |
| SSL on FAIR-CSAR tiles: teacher regions | 3pct | +0.5 / +0.5 / +0.0 | +0.4 | -1.0 / +0.2 / -0.4 | -0.4 |
| SSL on FAIR-CSAR tiles: teacher regions | 10pct | -0.0 / +0.2 / +0.7 | +0.3 | +0.1 / -0.1 / +1.6 | +0.5 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 1pct | +0.6 / +0.2 / +0.7 | +0.5 | +0.7 / +0.4 / +0.4 | +0.5 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 3pct | +0.8 / +0.0 / +1.3 | +0.7 | -0.2 / +0.2 / +0.9 | +0.3 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 10pct | -0.4 / +0.4 / -0.6 | -0.2 | -0.3 / +0.3 / +1.0 | +0.3 |

## Per-class accuracy on test_clean, mean of splits (%)


1pct:

| method | ship | aircraft | car | tank | bridge | harbor |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 96.7 | 92.2 | 82.9 | 64.4 | 96.0 | 53.6 |
| DINO on scenes | 97.9 | 92.3 | 87.5 | 62.4 | 95.2 | 58.1 |
| LeJEPA on scenes | 97.8 | 93.8 | 88.9 | 47.1 | 97.3 | 62.7 |
| SparK on scenes | 96.5 | 94.0 | 86.1 | 73.2 | 93.9 | 50.4 |
| RegionDINO scene+object | 97.8 | 92.8 | 87.3 | 65.8 | 96.4 | 54.2 |
| RegionDINO scene+object, random windows (control) | 96.5 | 93.4 | 85.9 | 76.2 | 90.5 | 59.2 |
| RegionDINO object-only | 97.5 | 93.7 | 86.9 | 63.6 | 95.7 | 57.1 |
| RegionDINO object-only, random windows (control) | 97.5 | 93.1 | 82.9 | 58.9 | 96.1 | 54.5 |
| RegionDINO object-only, teacher regions | 97.0 | 92.2 | 83.7 | 66.8 | 95.3 | 57.2 |
| RegionDINO object-only, teacher-size random windows (control) | 96.6 | 94.8 | 82.0 | 61.8 | 95.6 | 51.7 |
| SSL on FAIR-CSAR tiles: DINO | 97.7 | 92.2 | 87.6 | 61.8 | 95.7 | 51.7 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 97.5 | 93.2 | 85.2 | 69.6 | 94.3 | 55.1 |
| SSL on FAIR-CSAR tiles: random windows (control) | 97.2 | 94.5 | 87.2 | 62.4 | 95.5 | 54.8 |
| SSL on FAIR-CSAR tiles: teacher regions | 97.0 | 92.8 | 83.6 | 63.4 | 95.1 | 54.2 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 95.8 | 93.1 | 85.8 | 64.2 | 95.8 | 54.1 |

3pct:

| method | ship | aircraft | car | tank | bridge | harbor |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 98.4 | 97.5 | 96.1 | 88.3 | 97.0 | 71.3 |
| DINO on scenes | 99.0 | 97.4 | 97.0 | 86.6 | 94.8 | 77.8 |
| LeJEPA on scenes | 98.8 | 98.2 | 96.7 | 80.8 | 96.1 | 74.9 |
| SparK on scenes | 98.3 | 97.1 | 93.7 | 88.2 | 97.2 | 68.9 |
| RegionDINO scene+object | 98.5 | 97.7 | 97.3 | 84.0 | 95.5 | 75.7 |
| RegionDINO scene+object, random windows (control) | 98.3 | 97.4 | 94.1 | 89.1 | 95.4 | 73.2 |
| RegionDINO object-only | 98.6 | 97.3 | 96.8 | 89.4 | 96.6 | 72.5 |
| RegionDINO object-only, random windows (control) | 98.6 | 97.8 | 97.0 | 91.1 | 94.3 | 73.8 |
| RegionDINO object-only, teacher regions | 98.8 | 97.4 | 96.1 | 87.0 | 95.7 | 73.4 |
| RegionDINO object-only, teacher-size random windows (control) | 98.1 | 97.4 | 94.3 | 85.8 | 96.6 | 69.3 |
| SSL on FAIR-CSAR tiles: DINO | 98.3 | 97.6 | 95.2 | 91.4 | 93.6 | 80.6 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 98.0 | 96.7 | 90.7 | 88.5 | 94.4 | 75.5 |
| SSL on FAIR-CSAR tiles: random windows (control) | 98.6 | 96.9 | 95.1 | 87.3 | 96.5 | 74.9 |
| SSL on FAIR-CSAR tiles: teacher regions | 98.3 | 97.2 | 94.6 | 88.3 | 96.7 | 71.0 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 98.2 | 97.3 | 96.2 | 91.5 | 96.5 | 70.7 |

10pct:

| method | ship | aircraft | car | tank | bridge | harbor |
|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 98.8 | 99.2 | 99.0 | 97.3 | 97.4 | 87.3 |
| DINO on scenes | 99.6 | 99.9 | 99.5 | 96.1 | 97.7 | 89.0 |
| LeJEPA on scenes | 99.7 | 99.7 | 99.8 | 94.2 | 98.2 | 87.3 |
| SparK on scenes | 99.3 | 99.5 | 99.2 | 96.2 | 97.9 | 88.1 |
| RegionDINO scene+object | 99.6 | 99.4 | 99.9 | 95.2 | 98.1 | 90.3 |
| RegionDINO scene+object, random windows (control) | 99.6 | 99.8 | 99.7 | 96.0 | 98.2 | 90.1 |
| RegionDINO object-only | 99.1 | 99.8 | 99.6 | 95.8 | 97.9 | 88.4 |
| RegionDINO object-only, random windows (control) | 99.6 | 99.8 | 99.8 | 95.5 | 98.3 | 90.5 |
| RegionDINO object-only, teacher regions | 99.2 | 99.8 | 99.8 | 97.3 | 98.4 | 88.7 |
| RegionDINO object-only, teacher-size random windows (control) | 99.6 | 99.5 | 99.9 | 95.6 | 98.1 | 88.1 |
| SSL on FAIR-CSAR tiles: DINO | 99.4 | 99.4 | 99.4 | 97.0 | 98.0 | 90.2 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 99.4 | 99.6 | 99.6 | 96.8 | 98.1 | 87.0 |
| SSL on FAIR-CSAR tiles: random windows (control) | 99.1 | 99.5 | 99.6 | 96.7 | 97.9 | 87.6 |
| SSL on FAIR-CSAR tiles: teacher regions | 99.2 | 99.6 | 98.9 | 96.3 | 97.9 | 90.2 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 99.3 | 99.4 | 99.7 | 96.2 | 98.3 | 88.0 |

## Per-source accuracy on test_clean, mean of splits (%)


1pct:

| method | AIR_SARShip | HRSID | MSAR | OGSOD | SADD | SAR-AIRcraft | SIVED | SSDD | Shipdataset |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 80.2 | 93.2 | 88.3 | 82.5 | 97.5 | 95.3 | 82.9 | 92.8 | 99.2 |
| DINO on scenes | 89.3 | 95.8 | 89.3 | 81.6 | 97.3 | 94.9 | 87.5 | 93.1 | 99.7 |
| LeJEPA on scenes | 90.6 | 95.7 | 88.9 | 77.9 | 97.1 | 96.6 | 88.9 | 93.6 | 99.6 |
| SparK on scenes | 81.1 | 93.6 | 88.7 | 83.7 | 96.0 | 97.2 | 86.1 | 88.0 | 99.1 |
| RegionDINO scene+object | 90.0 | 95.2 | 89.7 | 83.3 | 97.4 | 95.5 | 87.3 | 91.1 | 99.6 |
| RegionDINO scene+object, random windows (control) | 87.1 | 91.9 | 89.1 | 83.6 | 96.6 | 96.1 | 85.9 | 90.0 | 99.4 |
| RegionDINO object-only | 89.2 | 94.2 | 90.4 | 82.2 | 97.1 | 96.3 | 86.9 | 91.3 | 99.7 |
| RegionDINO object-only, random windows (control) | 81.1 | 95.2 | 88.5 | 80.7 | 98.2 | 96.2 | 82.9 | 92.6 | 99.6 |
| RegionDINO object-only, teacher regions | 73.0 | 94.7 | 89.5 | 82.8 | 94.8 | 95.2 | 83.7 | 92.4 | 99.4 |
| RegionDINO object-only, teacher-size random windows (control) | 79.0 | 93.2 | 88.2 | 80.9 | 96.9 | 98.3 | 82.0 | 89.3 | 99.5 |
| SSL on FAIR-CSAR tiles: DINO | 86.5 | 95.3 | 89.1 | 81.0 | 97.0 | 95.3 | 87.6 | 93.3 | 99.5 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 83.2 | 95.2 | 88.2 | 83.3 | 95.8 | 96.7 | 85.2 | 94.2 | 99.4 |
| SSL on FAIR-CSAR tiles: random windows (control) | 79.3 | 95.2 | 88.7 | 81.3 | 97.4 | 97.9 | 87.2 | 90.6 | 99.2 |
| SSL on FAIR-CSAR tiles: teacher regions | 82.4 | 93.4 | 90.2 | 81.4 | 96.5 | 95.2 | 83.6 | 92.1 | 99.5 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 71.1 | 92.1 | 89.6 | 82.1 | 97.5 | 95.6 | 85.8 | 83.3 | 99.3 |

3pct:

| method | AIR_SARShip | HRSID | MSAR | OGSOD | SADD | SAR-AIRcraft | SIVED | SSDD | Shipdataset |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 90.4 | 96.6 | 93.1 | 92.3 | 98.9 | 99.9 | 96.1 | 92.7 | 99.8 |
| DINO on scenes | 94.7 | 98.2 | 93.6 | 90.6 | 99.1 | 99.5 | 97.0 | 97.0 | 99.8 |
| LeJEPA on scenes | 94.8 | 97.6 | 94.8 | 89.0 | 99.1 | 99.8 | 96.7 | 96.9 | 99.8 |
| SparK on scenes | 89.0 | 96.5 | 93.2 | 91.8 | 99.0 | 99.6 | 93.7 | 94.1 | 99.7 |
| RegionDINO scene+object | 90.8 | 97.1 | 93.7 | 89.9 | 98.9 | 99.8 | 97.3 | 95.7 | 99.7 |
| RegionDINO scene+object, random windows (control) | 90.1 | 96.7 | 92.7 | 91.6 | 99.1 | 99.9 | 94.1 | 95.2 | 99.7 |
| RegionDINO object-only | 93.2 | 97.0 | 93.2 | 92.3 | 99.1 | 99.9 | 96.8 | 94.9 | 99.8 |
| RegionDINO object-only, random windows (control) | 91.0 | 97.2 | 94.0 | 91.5 | 99.2 | 99.8 | 97.0 | 94.7 | 99.8 |
| RegionDINO object-only, teacher regions | 91.1 | 97.6 | 92.9 | 91.0 | 99.3 | 99.9 | 96.1 | 96.4 | 99.9 |
| RegionDINO object-only, teacher-size random windows (control) | 89.9 | 97.0 | 92.8 | 90.6 | 99.1 | 99.7 | 94.3 | 90.2 | 99.5 |
| SSL on FAIR-CSAR tiles: DINO | 86.9 | 96.8 | 94.1 | 91.7 | 99.2 | 99.5 | 95.2 | 94.3 | 99.8 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 80.7 | 96.4 | 93.9 | 90.5 | 97.8 | 98.8 | 90.7 | 92.3 | 99.8 |
| SSL on FAIR-CSAR tiles: random windows (control) | 88.9 | 97.5 | 92.4 | 91.6 | 98.5 | 99.7 | 95.1 | 95.0 | 99.8 |
| SSL on FAIR-CSAR tiles: teacher regions | 86.5 | 96.9 | 94.1 | 91.8 | 98.1 | 99.2 | 94.6 | 93.9 | 99.7 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 87.7 | 96.5 | 93.2 | 92.7 | 98.7 | 99.8 | 96.2 | 91.2 | 99.8 |

10pct:

| method | AIR_SARShip | HRSID | MSAR | OGSOD | SADD | SAR-AIRcraft | SIVED | SSDD | Shipdataset |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet start (no SSL) | 98.7 | 97.1 | 98.5 | 96.5 | 99.3 | 99.5 | 99.0 | 94.4 | 99.7 |
| DINO on scenes | 99.3 | 99.1 | 99.3 | 96.4 | 99.5 | 99.9 | 99.5 | 99.0 | 99.9 |
| LeJEPA on scenes | 98.9 | 99.4 | 98.9 | 95.9 | 99.4 | 99.9 | 99.8 | 99.1 | 100.0 |
| SparK on scenes | 97.6 | 98.2 | 99.1 | 96.5 | 99.3 | 99.6 | 99.2 | 97.9 | 99.9 |
| RegionDINO scene+object | 98.9 | 99.0 | 98.2 | 96.5 | 99.4 | 99.9 | 99.9 | 98.4 | 99.9 |
| RegionDINO scene+object, random windows (control) | 98.9 | 99.3 | 99.1 | 96.7 | 99.6 | 99.9 | 99.7 | 99.1 | 99.9 |
| RegionDINO object-only | 98.6 | 97.7 | 99.1 | 96.4 | 99.4 | 99.9 | 99.6 | 96.7 | 99.9 |
| RegionDINO object-only, random windows (control) | 98.6 | 98.9 | 99.2 | 96.7 | 99.4 | 99.9 | 99.8 | 98.4 | 99.9 |
| RegionDINO object-only, teacher regions | 98.5 | 97.9 | 99.2 | 97.3 | 99.4 | 99.9 | 99.8 | 97.6 | 99.9 |
| RegionDINO object-only, teacher-size random windows (control) | 98.2 | 99.4 | 98.5 | 96.5 | 99.6 | 99.9 | 99.9 | 98.5 | 99.9 |
| SSL on FAIR-CSAR tiles: DINO | 99.2 | 98.7 | 99.2 | 97.0 | 99.5 | 99.4 | 99.4 | 98.1 | 99.9 |
| SSL on FAIR-CSAR tiles: RegionDINO object-only | 98.5 | 98.7 | 98.9 | 96.7 | 99.2 | 99.8 | 99.6 | 97.0 | 99.8 |
| SSL on FAIR-CSAR tiles: random windows (control) | 98.1 | 97.4 | 98.5 | 96.6 | 99.4 | 99.9 | 99.6 | 98.1 | 99.9 |
| SSL on FAIR-CSAR tiles: teacher regions | 99.0 | 97.8 | 98.7 | 96.8 | 99.3 | 99.9 | 98.9 | 98.6 | 99.8 |
| SSL on FAIR-CSAR tiles: teacher-size random windows (control) | 98.6 | 98.3 | 98.4 | 96.7 | 99.4 | 99.9 | 99.7 | 97.8 | 99.9 |
