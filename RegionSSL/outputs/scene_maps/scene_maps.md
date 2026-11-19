# Frozen backbones on SARDet-100K test scenes: where the response sits, before fine-tuning

1500 official test scenes with at least one annotated object (seed 0, the same scenes for every model and split), shorter side 512 px as in the SSL views and the detector; boxes = the dataset's annotations, mean area share 3.4 % of the scene (what a uniform map scores as mass in boxes). Response map = channel norm of the last feature map of the frozen backbone (stride 32), no labels, no fine-tuning. Concentration = mass in boxes / area share. Peak in boxes = share of scenes whose strongest pixel lies inside a box.

## Split 1

| backbone | mass in boxes (%) | concentration | peak in boxes (%) | vs ImageNet start: mean, % scenes higher | vs DINO on SARDet: mean, % scenes higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 6.5 | 2.54 | 36.4 | - | +2.4, 87 % |
| DINO on SARDet | 4.0 | 1.32 | 7.6 | -2.4, 13 % | - |
| LeJEPA on SARDet | 3.6 | 1.09 | 8.5 | -2.8, 12 % | -0.4, 35 % |
| SparK on SARDet | 5.5 | 2.09 | 26.8 | -1.0, 29 % | +1.5, 62 % |
| RegionDINO scene+object | 4.1 | 1.35 | 17.3 | -2.4, 12 % | +0.1, 56 % |
| RegionDINO scene+object, random windows | 3.8 | 1.23 | 13.1 | -2.7, 11 % | -0.2, 42 % |
| RegionDINO object-only | 4.2 | 1.39 | 28.7 | -2.2, 19 % | +0.2, 60 % |
| RegionDINO object-only, random windows | 4.1 | 1.44 | 19.7 | -2.4, 14 % | +0.1, 56 % |

## Split 2

| backbone | mass in boxes (%) | concentration | peak in boxes (%) | vs ImageNet start: mean, % scenes higher | vs DINO on SARDet: mean, % scenes higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 6.5 | 2.54 | 36.4 | - | +2.3, 85 % |
| DINO on SARDet | 4.2 | 1.33 | 11.1 | -2.3, 15 % | - |
| LeJEPA on SARDet | 3.7 | 1.08 | 7.8 | -2.8, 13 % | -0.5, 28 % |
| SparK on SARDet | 7.0 | 2.99 | 32.7 | +0.6, 38 % | +2.9, 72 % |
| RegionDINO scene+object | 4.0 | 1.32 | 17.5 | -2.5, 12 % | -0.2, 48 % |
| RegionDINO scene+object, random windows | 3.7 | 1.19 | 8.1 | -2.7, 10 % | -0.5, 30 % |
| RegionDINO object-only | 4.4 | 1.47 | 33.3 | -2.1, 18 % | +0.2, 62 % |
| RegionDINO object-only, random windows | 3.5 | 1.31 | 15.1 | -2.9, 11 % | -0.7, 44 % |

## Split 3

| backbone | mass in boxes (%) | concentration | peak in boxes (%) | vs ImageNet start: mean, % scenes higher | vs DINO on SARDet: mean, % scenes higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 6.5 | 2.54 | 36.4 | - | +2.2, 85 % |
| DINO on SARDet | 4.2 | 1.40 | 10.6 | -2.2, 15 % | - |
| LeJEPA on SARDet | 3.7 | 1.10 | 7.5 | -2.7, 13 % | -0.5, 26 % |
| SparK on SARDet | 6.1 | 2.46 | 33.4 | -0.3, 37 % | +1.9, 73 % |
| RegionDINO scene+object | 4.0 | 1.35 | 21.3 | -2.4, 11 % | -0.2, 41 % |
| RegionDINO scene+object, random windows | 4.0 | 1.26 | 13.5 | -2.5, 14 % | -0.3, 38 % |
| RegionDINO object-only | 4.5 | 1.50 | 31.3 | -2.0, 18 % | +0.3, 59 % |
| RegionDINO object-only, random windows | 4.6 | 1.72 | 27.5 | -1.9, 16 % | +0.4, 63 % |

## Per source dataset (all splits pooled): mass in boxes % / peak in boxes %

| source | n scenes | boxes area % | ImageNet start (no SSL) | DINO on SARDet | LeJEPA on SARDet | SparK on SARDet | RegionDINO scene+object | RegionDINO scene+object, random windows | RegionDINO object-only | RegionDINO object-only, random windows |
|---|---|---|---|---|---|---|---|---|---|---|
| AIR_SARShip | 14 | 6.2 | 11.3 / 57 | 9.9 / 36 | 7.3 / 17 | 11.2 / 31 | 8.9 / 33 | 9.2 / 33 | 8.6 / 36 | 8.1 / 36 |
| HRSID | 133 | 1.1 | 2.6 / 24 | 1.5 / 11 | 1.2 / 2 | 2.1 / 37 | 1.6 / 21 | 1.5 / 10 | 1.5 / 36 | 1.7 / 24 |
| MSAR | 202 | 3.7 | 6.7 / 37 | 4.4 / 9 | 3.9 / 10 | 6.1 / 33 | 4.2 / 13 | 3.9 / 7 | 4.2 / 30 | 4.1 / 17 |
| OGSOD | 214 | 2.9 | 4.1 / 9 | 2.9 / 2 | 3.0 / 2 | 3.2 / 2 | 2.9 / 2 | 3.0 / 3 | 3.1 / 3 | 2.8 / 2 |
| SADD | 4 | 5.9 | 7.8 / 25 | 6.8 / 8 | 6.7 / 0 | 6.7 / 0 | 7.3 / 8 | 6.0 / 17 | 8.5 / 33 | 7.2 / 8 |
| SAR-AIRcraft | 371 | 5.5 | 8.0 / 21 | 6.8 / 8 | 6.6 / 16 | 6.3 / 2 | 6.3 / 12 | 6.3 / 11 | 7.3 / 29 | 5.8 / 12 |
| SIVED | 12 | 8.9 | 12.1 / 25 | 12.2 / 33 | 10.9 / 19 | 9.4 / 3 | 10.8 / 36 | 10.2 / 31 | 13.5 / 42 | 10.2 / 19 |
| SSDD | 10 | 1.5 | 4.4 / 40 | 2.4 / 10 | 2.0 / 20 | 2.6 / 23 | 2.4 / 27 | 2.1 / 10 | 2.7 / 50 | 2.3 / 10 |
| Shipdataset | 540 | 2.4 | 7.0 / 60 | 3.1 / 13 | 2.2 / 5 | 8.3 / 62 | 3.2 / 31 | 2.8 / 17 | 3.3 / 42 | 3.7 / 35 |

Written 2026-11-19T12:22:57.
