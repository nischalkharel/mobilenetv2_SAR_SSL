# Map mass on the vehicle box: A1 versus P4 (split 1), random test chips

2000 official test chips (seed 0); box = the dataset's vehicle annotation, mean box area 17.2 % of the chip (what a uniform map would score). Grad-CAM: 1 % fine-tuned model, true class.
Response map: frozen SSL backbone, channel norm of the last feature map. Concentration = mass share / box area share.

| Map | Mass in box (%) | Concentration on vehicle | Accuracy on these chips (%) |
|---|---|---|---|
| A1 fine-tuned 1 % | 37.1 | 2.32 | 35.6 |
| A1 backbone | 34.1 | 2.25 | - |
| P4 fine-tuned 1 % | 39.2 | 2.48 | 39.9 |
| P4 backbone | 36.2 | 2.40 | - |

fine-tuned 1 %: P4 minus A1 mass in box, mean +1.9 points; P4 higher on 52 % of chips.

backbone: P4 minus A1 mass in box, mean +2.1 points; P4 higher on 83 % of chips.

Written 2026-11-11T14:28:16.
