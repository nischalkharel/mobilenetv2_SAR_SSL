# Frozen backbones on ATRNet-STAR test chips: where the response sits, before fine-tuning

2000 official test chips (seed 0, the same chips for every model and split); box = the dataset's vehicle annotation, mean box area 17.2 % of the chip (what a uniform map scores as mass in box). Response map = channel norm of the last feature map of the frozen backbone, no labels, no fine-tuning. Concentration = mass in box / box area share. Peak in box = share of chips whose strongest pixel lies in the box.

## Split 1

| backbone | mass in box (%) | concentration | peak in box (%) | vs ImageNet start: mean, % chips higher | vs XA DINO on SARDet: mean, % chips higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 25.7 | 1.60 | 47.4 | - | +7.2, 93 % |
| A1 DINO on ATRNet (in-domain, split 1) | 34.1 | 2.25 | 87.4 | +8.4, 97 % | +15.6, 100 % |
| XA DINO on SARDet | 18.6 | 1.07 | 17.8 | -7.2, 7 % | - |
| XA LeJEPA on SARDet | 21.0 | 1.26 | 28.6 | -4.8, 17 % | +2.4, 79 % |
| XA SparK on SARDet | 18.8 | 1.11 | 16.0 | -6.9, 4 % | +0.3, 54 % |
| RegionDINO scene+object | 18.4 | 1.07 | 11.6 | -7.3, 4 % | -0.1, 49 % |
| RegionDINO scene+object, random windows | 14.9 | 0.82 | 11.0 | -10.8, 2 % | -3.6, 8 % |
| RegionDINO object-only | 28.0 | 1.83 | 62.1 | +2.3, 65 % | +9.5, 94 % |
| RegionDINO object-only, random windows | 18.3 | 1.09 | 13.1 | -7.4, 7 % | -0.2, 49 % |

## Split 2

| backbone | mass in box (%) | concentration | peak in box (%) | vs ImageNet start: mean, % chips higher | vs XA DINO on SARDet: mean, % chips higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 25.7 | 1.60 | 47.4 | - | +6.6, 91 % |
| XA DINO on SARDet | 19.1 | 1.11 | 17.2 | -6.6, 9 % | - |
| XA LeJEPA on SARDet | 19.1 | 1.13 | 21.1 | -6.6, 10 % | +0.0, 54 % |
| XA SparK on SARDet | 18.6 | 1.11 | 13.9 | -7.1, 4 % | -0.5, 47 % |
| RegionDINO scene+object | 18.9 | 1.09 | 21.0 | -6.9, 6 % | -0.3, 46 % |
| RegionDINO scene+object, random windows | 16.2 | 0.91 | 12.7 | -9.5, 2 % | -2.9, 8 % |
| RegionDINO object-only | 25.9 | 1.68 | 56.2 | +0.2, 53 % | +6.8, 88 % |
| RegionDINO object-only, random windows | 16.4 | 0.96 | 13.8 | -9.4, 6 % | -2.8, 30 % |

## Split 3

| backbone | mass in box (%) | concentration | peak in box (%) | vs ImageNet start: mean, % chips higher | vs XA DINO on SARDet: mean, % chips higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 25.7 | 1.60 | 47.4 | - | +5.9, 90 % |
| XA DINO on SARDet | 19.8 | 1.16 | 19.4 | -5.9, 10 % | - |
| XA LeJEPA on SARDet | 23.3 | 1.40 | 31.4 | -2.5, 31 % | +3.5, 85 % |
| XA SparK on SARDet | 18.9 | 1.12 | 15.2 | -6.9, 5 % | -0.9, 40 % |
| RegionDINO scene+object | 18.9 | 1.10 | 14.1 | -6.8, 7 % | -0.9, 34 % |
| RegionDINO scene+object, random windows | 16.3 | 0.91 | 10.2 | -9.5, 4 % | -3.5, 7 % |
| RegionDINO object-only | 26.3 | 1.71 | 59.2 | +0.6, 55 % | +6.5, 86 % |
| RegionDINO object-only, random windows | 19.4 | 1.17 | 19.1 | -6.3, 9 % | -0.4, 49 % |

Written 2026-11-18T13:46:25.
