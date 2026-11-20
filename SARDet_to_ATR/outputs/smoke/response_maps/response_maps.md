# Frozen backbones on ATRNet-STAR test chips: where the response sits, before fine-tuning

64 official test chips (seed 0, the same chips for every model and split); box = the dataset's vehicle annotation, mean box area 19.2 % of the chip (what a uniform map scores as mass in box). Response map = channel norm of the last feature map of the frozen backbone, no labels, no fine-tuning. Concentration = mass in box / box area share. Peak in box = share of chips whose strongest pixel lies in the box.

## Split 1

| backbone | mass in box (%) | concentration | peak in box (%) | vs ImageNet start: mean, % chips higher | vs XA DINO on SARDet: mean, % chips higher |
|---|---|---|---|---|---|
| ImageNet start (no SSL) | 27.1 | 1.60 | 57.8 | - | - |
| RegionDINO smoke (2 epochs) | 26.1 | 1.56 | 35.9 | -1.1, 47 % | - |

Written 2026-11-18T09:14:08.
